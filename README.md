Rainbet Blackjack Assistant
===========================

This is a small macOS helper that watches a fullscreen Rainbet blackjack table, reads the cards from the screen using OCR, and speaks basic‑strategy advice.

How it works
------------

- Captures the chosen screen using `mss`.
- Detects the bright white card rectangles on the table with OpenCV.
- Splits them into a top row (dealer) and bottom row (player).
- OCRs the rank on each card using Tesseract.
- Computes player total / soft vs hard, plus dealer upcard value.
- Applies multi‑deck, dealer‑stands‑on‑soft‑17 basic strategy (no splits).
- Speaks the recommended move via macOS `say`.
- Only speaks when the **player hand changes** and the player total is **≤ 21**.

Requirements
-----------

- macOS (uses the built‑in `say` command).
- Python 3.11+.

Install dependencies:

```bash
uv sync
```

Usage
-----

Run the assistant:

```bash
uv run main.py
```

Command‑line options:

- `-v`, `--verbose-speech`  
  Speak “player X [soft], dealer Y. move” instead of just “hit/stand/double”.

- `-d`, `--debug`  
  Enable debug logging (OCR output, detection info).

Examples:

```bash
# Normal, concise advice
uv run main.py

# Verbose voice with scores
uv run main.py -v

# Verbose voice + debug logging
uv run main.py -v -d
```

On startup the program lists available screens (as reported by `mss`); choose the one where Safari with Rainbet blackjack is fullscreen. No separate calibration step is needed.

Notes
-----

- The assistant does not keep a running/true count; it only plays basic strategy based on the current hand.
- Splits are not implemented because we do not track exact card pairs beyond what’s needed for totals.
