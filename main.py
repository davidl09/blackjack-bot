import subprocess
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from mss import mss

DEBUG = False
VERBOSE_SPEECH = False


def debug(msg: str) -> None:
    if DEBUG:
        print(msg)


def choose_monitor_index() -> int:
    """
    List available screens and let the user pick one.
    Indexing follows mss: 1..N are real monitors.
    """
    with mss() as sct:
        monitors = sct.monitors

    print("\nAvailable screens:")
    for idx, m in enumerate(monitors[1:], start=1):
        print(f"  {idx}: {m['width']}x{m['height']} at ({m['left']},{m['top']})")

    while True:
        choice = input("Select screen number to watch (e.g. 1): ").strip()
        if not choice.isdigit():
            print("Please enter a number like 1 or 2.")
            continue
        idx = int(choice)
        if 1 <= idx < len(monitors):
            return idx
        print("Invalid screen index, try again.")


def grab_monitor(sct: mss, monitor_index: int) -> np.ndarray:
    """Grab a full screenshot of the chosen monitor."""
    monitor = sct.monitors[monitor_index]
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def detect_card_rects(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect bright white card rectangles on the table.
    Returns a list of bounding boxes (x, y, w, h) in screen coordinates.
    """
    h, w, _ = frame.shape

    # Focus on the central band where the cards live to
    # avoid bright UI elements on the edges.
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    roi = frame[:, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold for bright rectangles (cards are white on dark blue background).
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rects: List[Tuple[int, int, int, int]] = []
    frame_area = float(h * w)
    min_area = frame_area * 0.0008
    max_area = frame_area * 0.03

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        if area < min_area or area > max_area:
            continue

        aspect = h_box / float(w_box) if w_box > 0 else 0.0
        # Cards are vertical rectangles, roughly 1.2–1.8 aspect ratio.
        if not (1.1 <= aspect <= 1.9):
            continue

        mean_val = cv2.mean(gray[y : y + h_box, x : x + w_box])[0]
        if mean_val < 200:
            continue

        # Map back into full-frame coordinates.
        rects.append((x + x1, y, w_box, h_box))

    return rects


def ocr_card_rank(card_img: np.ndarray) -> Optional[str]:
    """
    OCR the rank shown on a single card.
    Returns a rank string like '2'..'10', 'A', 'J', 'Q', 'K'.
    """
    h, w, _ = card_img.shape

    def ocr_on_roi(y0: int, y1: int, x0: int, x1: int, tag: str) -> Optional[str]:
        roi = card_img[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Try both normal and inverted Otsu thresholding.
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        inv = cv2.bitwise_not(thresh)

        config = (
            r"--oem 3 --psm 7 "
            r"-c tessedit_char_whitelist=0123456789AJQKT"
        )

        for img, variant in ((thresh, "norm"), (inv, "inv")):
            raw = pytesseract.image_to_string(img, config=config)
            text = raw.strip().upper().replace(" ", "")
            debug(f"[CARD OCR:{tag}:{variant}] raw={raw!r} cleaned='{text}'")
            if text:
                return text
        return None

    # First pass: tight region around upper-middle where rank usually sits.
    top1 = int(h * 0.08)
    bottom1 = int(h * 0.45)
    left1 = int(w * 0.15)
    right1 = int(w * 0.85)
    text = ocr_on_roi(top1, bottom1, left1, right1, "tight")

    # Second pass: slightly larger region around the center, useful when
    # cards overlap and the rank is shifted.
    if not text:
        top2 = int(h * 0.05)
        bottom2 = int(h * 0.65)
        left2 = int(w * 0.05)
        right2 = int(w * 0.95)
        text = ocr_on_roi(top2, bottom2, left2, right2, "wide")

    if not text:
        return None

    # Handle obvious 10 variants first.
    if "10" in text:
        return "10"
    if text.startswith("1") and len(text) >= 2 and text[1] == "0":
        return "10"

    # Single-rank letters.
    if any(ch in text for ch in ("A", "J", "Q", "K", "T")):
        for ch in ("A", "J", "Q", "K", "T"):
            if ch in text:
                return "10" if ch == "T" else ch

    # Digits 2–9 (ignore stray extra characters).
    digits = [ch for ch in text if ch.isdigit()]
    if digits:
        # If we somehow see "11" treat as 11 (A shown weirdly),
        # otherwise just use the first digit.
        s = "".join(digits)
        if s == "11":
            return "A"
        if len(s) == 1:
            return s
        if len(s) == 2 and s[0] == "1" and s[1] == "0":
            return "10"
        return digits[0]

    return None


def card_values_from_ranks(ranks: List[str]) -> Tuple[int, bool]:
    """
    Compute total and softness from a list of ranks.
    Returns (total, is_soft).
    """
    total = 0
    aces = 0

    for rank in ranks:
        if rank == "A":
            aces += 1
            total += 11
        elif rank in {"J", "Q", "K", "10"}:
            total += 10
        else:
            try:
                total += int(rank)
            except ValueError:
                continue

    soft = False
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    if aces > 0 and total <= 21:
        soft = True

    return total, soft


def basic_strategy_from_scores(
    player_total: int, player_soft: bool, dealer_total: int
) -> str:
    """
    Basic strategy that only uses totals and whether the player has a soft hand.
    We ignore pair-specific split logic because we no longer know the exact cards.
    """
    dealer_val = dealer_total

    # Soft hands
    if player_soft:
        if player_total >= 19:
            return "stand"
        if player_total == 18:
            if dealer_val in {2, 7, 8}:
                return "stand"
            if 3 <= dealer_val <= 6:
                return "double"
            return "hit"
        if player_total == 17:
            if 3 <= dealer_val <= 6:
                return "double"
            return "hit"
        if player_total in {15, 16}:
            if 4 <= dealer_val <= 6:
                return "double"
            return "hit"
        if player_total in {13, 14}:
            if 5 <= dealer_val <= 6:
                return "double"
            return "hit"

    # Hard hands
    if player_total >= 17:
        return "stand"
    if 13 <= player_total <= 16:
        if 2 <= dealer_val <= 6:
            return "stand"
        return "hit"
    if player_total == 12:
        if 4 <= dealer_val <= 6:
            return "stand"
        return "hit"
    if player_total == 11:
        # S17: double 11 vs 2–10, HIT vs dealer Ace
        if dealer_val == 11:
            return "hit"
        return "double"
    if player_total == 10:
        if 2 <= dealer_val <= 9:
            return "double"
        return "hit"
    if player_total == 9:
        if 3 <= dealer_val <= 6:
            return "double"
        return "hit"

    return "hit"


def speak(move: str) -> None:
    # macOS text-to-speech
    # Use a faster speaking rate for snappier feedback.
    subprocess.run(["say", "-r", "260", move], check=False)


def main() -> None:
    monitor_index = choose_monitor_index()
    sct = mss()
    last_player_state: Optional[Tuple[Tuple[str, ...], int, bool]] = None

    print(
        "Rainbet blackjack assistant (card-based) running. "
        "Press CTRL+C to stop."
    )

    try:
        while True:
            frame = grab_monitor(sct, monitor_index)

            # Detect card rectangles and split into top (dealer) and bottom (player) rows.
            rects = detect_card_rects(frame)
            if len(rects) < 2:
                time.sleep(0.2)
                continue

            # Sort by vertical position, then split around the midpoint.
            centers = [(y + h // 2, (x, y, w, h)) for (x, y, w, h) in rects]
            centers.sort(key=lambda t: t[0])
            ys = [c for c, _ in centers]
            mid_y = (min(ys) + max(ys)) / 2.0

            dealer_rects = [rect for (cy, rect) in centers if cy < mid_y]
            player_rects = [rect for (cy, rect) in centers if cy >= mid_y]

            dealer_rects.sort(key=lambda r: r[0])  # left to right
            player_rects.sort(key=lambda r: r[0])

            if not player_rects or not dealer_rects:
                time.sleep(0.2)
                continue

            # OCR ranks for each card.
            dealer_ranks: List[str] = []
            for x, y, w_box, h_box in dealer_rects:
                card_img = frame[y : y + h_box, x : x + w_box]
                rank = ocr_card_rank(card_img)
                if rank is not None:
                    dealer_ranks.append(rank)

            player_ranks: List[str] = []
            for x, y, w_box, h_box in player_rects:
                card_img = frame[y : y + h_box, x : x + w_box]
                rank = ocr_card_rank(card_img)
                if rank is not None:
                    player_ranks.append(rank)

            if not player_ranks or not dealer_ranks:
                time.sleep(0.2)
                continue

            # Use dealer upcard only (first visible card).
            dealer_up_rank = dealer_ranks[0]
            dealer_total, _dealer_soft = card_values_from_ranks([dealer_up_rank])

            player_total, player_soft = card_values_from_ranks(player_ranks)

            # Only react when the PLAYER hand changes (ranks/total/softness).
            player_state = (tuple(player_ranks), player_total, player_soft)
            if player_state != last_player_state:
                last_player_state = player_state

                move = basic_strategy_from_scores(
                    player_total, player_soft, dealer_total
                )

                label_soft = " (soft)" if player_soft else ""
                print(
                    f"Player cards: {player_ranks} -> {player_total}{label_soft}  "
                    f"Dealer upcard: {dealer_up_rank} ({dealer_total})  -> {move.upper()}"
                )
                # Speak only when:
                #   - player has at least two cards (initial deal complete)
                #   - total <= 21 (no advice after bust)
                if len(player_ranks) >= 2 and player_total <= 21:
                    if VERBOSE_SPEECH:
                        phrase = (
                            f"player {player_total}{' soft' if player_soft else ''}, "
                            f"dealer {dealer_total}. {move}"
                        )
                        speak(phrase)
                    else:
                        speak(move)

            # Faster polling; we only speak on player changes anyway.
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting assistant.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rainbet blackjack assistant"
    )
    parser.add_argument(
        "-v",
        "--verbose-speech",
        action="store_true",
        help="Speak player and dealer scores along with the move",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging and OCR output",
    )
    args = parser.parse_args()

    DEBUG = args.debug
    VERBOSE_SPEECH = args.verbose_speech

    main()
