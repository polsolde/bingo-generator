# Bingo Generator

Generate unique 90-ball bingo cards and export them to PDF with a configurable multi-card page layout.

## Features

- 90-ball bingo card generation with uniqueness tracking.
- Landscape/portrait page support (currently configured as landscape A4).
- Configurable page grid (`columns x rows`), margins, and inter-card spacing.
- Per-card header line: `RONDA X` (left) and event title (right).
- Optional image for empty cells with center-crop + resize preserving target aspect ratio.
- Configurable number style (including bold), colors, borders, and font sizes.
- Layout-fit validation before rendering to prevent broken pagination.

## Requirements

- Python 3.10+ (recommended)
- Dependencies from `requirements.txt`:
  - `pillow`
  - `reportlab`

## Setup (WSL + venv)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./venv/bin/python generate_pdf.py
```

Output:
- `my_bingo_cards.pdf`
- Console summary with active configuration and computed cell size.

## Configuration

All configuration lives in `BingoConfig` inside `generate_pdf.py`.

### Page layout

- `page_columns`: number of card columns per page.
- `page_rows`: number of card rows per page.
- `page_size`: ReportLab page size tuple (default `landscape(A4)`).
- `page_margin_top`, `page_margin_bottom`, `page_margin_left`, `page_margin_right`: page margins in mm.
- `card_gap`: horizontal and vertical spacing between cards in mm.

### Card content/layout

- `round_number`: used in `RONDA {round_number}`.
- `card_width`, `card_height`: full card block size in mm (header + bingo table).
- `header_gap`: spacing between header line and bingo grid in mm.
- `event_title`: right header text.
- `round_font_size`, `event_title_font_size`: header font sizes.
- `empty_image_path`: optional path used for empty cells.

### Table style

- `font_size`: bingo number font size.
- `number_bold`: `True` for bold numbers.
- `number_color`: bingo number color (hex).
- `empty_color`: fallback empty-cell background color (used when no image).
- `header_text_color`: color for `RONDA` and event title.
- `border_color`, `border_width`: grid border styling.

### Derived values

- `cards_per_page` = `page_columns * page_rows`.
- `get_cell_size_mm()` returns real bingo cell size after header/table split.

## Current defaults

The project is currently tuned for:
- 2 columns x 3 rows per page
- A4 landscape
- Slim horizontal margins
- Card header on one line with bold-ish visual emphasis
- Number bold enabled by default

Check the `BingoConfig` class for exact current values.

## Project structure

- `generate_pdf.py`: generator, PDF renderer, and probability utilities.
- `requirements.txt`: runtime dependencies.
- `my_bingo_cards.pdf`: generated output (example artifact).

## Next updates

There are a few updates planned for a future:

1. **Split modules by responsibility**
   - Move `BingoCard`, `BingoCardGenerator`, and `ProbabilityCalculator` into separate files.
2. **Introduce a CLI**
   - Add argparse (`--pages`, `--output`, `--round`, `--config-file`) to avoid editing source for runs.
3. **External config file**
   - Load JSON/YAML into `BingoConfig` for easier event-by-event customization.
4. **Logging instead of prints**
   - Replace direct `print()` calls with structured logging.
5. **Tests**
   - Unit tests for card validity constraints and layout-fit behavior.
6. **Performance for large runs**
   - Consider batched card generation and optional multiprocessing for very high card counts.

## Notes

- If `empty_image_path` is invalid or missing, empty cells gracefully fallback to background color.
- If layout does not fit, generation raises a clear `ValueError` with guidance.