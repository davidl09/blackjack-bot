import json

import cv2
import numpy as np
from mss import mss


def choose_monitor_index() -> int:
    """
    List available screens and let the user pick one.
    Indexing follows mss: 1..N are real monitors.
    """
    with mss() as sct:
        monitors = sct.monitors
    print("\nAvailable screens:")
    for idx, m in enumerate(monitors[1:], start=1):
        print(
            f"  {idx}: {m['width']}x{m['height']} at ({m['left']},{m['top']})"
        )
    while True:
        choice = input("Select screen number to calibrate (e.g. 1): ").strip()
        if not choice.isdigit():
            print("Please enter a number like 1 or 2.")
            continue
        idx = int(choice)
        if 1 <= idx < len(monitors):
            return idx
        print("Invalid screen index, try again.")


def select_score_boxes(monitor_index: int) -> dict:
    """
    Take a single screenshot of the chosen screen and let the user
    draw TWO rectangles on it: dealer score and player score.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        raw = sct.grab(monitor)
        img = np.array(raw)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    window_name = "Select dealer and player scores"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nInstructions:")
    print("  1) Draw a rectangle around the DEALER score, then press ENTER.")
    print("  2) Draw a rectangle around the PLAYER score, then press ENTER again.")
    print("     - Click and drag with the left mouse button to draw.")
    print("     - Release to finish drawing.")
    print("     - Press 'r' to clear the current selection and redraw.")
    print("     - Press ESC or 'q' to cancel.\n")

    drawing = False
    start_x = 0
    start_y = 0
    current_rect = None  # (x, y, w, h)
    dealer_rect = None
    player_rect = None
    phase = 0  # 0 = selecting dealer, 1 = selecting player

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_x, start_y, current_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y
            current_rect = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, y1 = min(start_x, x), min(start_y, y)
            x2, y2 = max(start_x, x), max(start_y, y)
            current_rect = (x1, y1, x2 - x1, y2 - y1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = min(start_x, x), min(start_y, y)
            x2, y2 = max(start_x, x), max(start_y, y)
            current_rect = (x1, y1, x2 - x1, y2 - y1)

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = img.copy()
        if dealer_rect is not None:
            x, y, w, h = dealer_rect
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if player_rect is not None:
            x, y, w, h = player_rect
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if current_rect is not None:
            x, y, w, h = current_rect
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # ENTER
            if current_rect is None:
                continue
            if phase == 0:
                dealer_rect = current_rect
                current_rect = None
                phase = 1
                print("Dealer score selected. Now select PLAYER score and press ENTER.")
            else:
                player_rect = current_rect
                current_rect = None
                break
        elif key in (27, ord("q")):  # ESC or q
            dealer_rect = None
            player_rect = None
            break
        elif key in (ord("r"), ord("R")):
            current_rect = None

    cv2.destroyWindow(window_name)

    if dealer_rect is None or player_rect is None:
        raise RuntimeError("Calibration cancelled.")

    dx, dy, dw, dh = dealer_rect
    px, py, pw, ph = player_rect

    dealer_region = {
        "top": int(monitor["top"] + dy),
        "left": int(monitor["left"] + dx),
        "width": int(dw),
        "height": int(dh),
    }
    player_region = {
        "top": int(monitor["top"] + py),
        "left": int(monitor["left"] + px),
        "width": int(pw),
        "height": int(ph),
    }

    print(f"Dealer score region -> {dealer_region}")
    print(f"Player score region -> {player_region}")

    return {
        "dealer_score_region": dealer_region,
        "player_score_region": player_region,
    }


def main() -> None:
    print("\n=== RAINBET BLACKJACK SCORE CALIBRATION ===")
    print("You will mark two small boxes on a specific screen:")
    print("  1) Dealer SCORE (the number shown under dealer cards)")
    print("  2) Player SCORE (the number shown under your cards)")
    print("\nEach box should tightly cover just the score text.")

    monitor_index = choose_monitor_index()
    points = select_score_boxes(monitor_index)

    with open("regions.json", "w", encoding="utf-8") as f:
        json.dump(points, f, indent=4)

    print("\nCalibration saved to regions.json ✅")


if __name__ == "__main__":
    main()
