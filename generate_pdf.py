#!/usr/bin/env python3
"""
90-Ball Bingo Card Generator
A highly customizable bingo card generator with PDF output and probability calculations.
"""

import random
import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
import io
import os
from pprint import pformat

@dataclass
class BingoConfig:
    """Configuration for bingo generation and PDF rendering.

    Units:
    - All sizes/margins/gaps are millimeters.
    """
    # Page layout
    pages: int = 3
    page_columns: int = 2
    page_rows: int = 3
    page_size = landscape(A4)
    page_margin_top: float = 2
    page_margin_bottom: float = 0
    page_margin_left: float = 2
    page_margin_right: float = 0
    card_gap: float = 5  # same spacing for horizontal and vertical separation

    # Card layout/content
    round_number: int = 1
    card_width: float = 144
    card_height: float = 65
    header_gap: float = 0.0
    event_title: str = "GRAN BINGO DELS 25 ANYS DE FES-TE JOVE!"
    round_font_size: int = 12
    event_title_font_size: int = 12
    empty_image_path: Optional[str] = "Logo_FJ_imatge-invert.jpg"

    # Table visual style
    font_size: int = 24
    number_bold: bool = True
    number_color: str = '#000000'
    empty_color: str = '#E0E0E0'
    header_text_color: str = '#000000'
    border_color: str = '#000000'
    border_width: float = 1

    @property
    def cards_per_page(self) -> int:
        return self.page_columns * self.page_rows

    def __post_init__(self) -> None:
        positive_ints = {
            "pages": self.pages,
            "page_columns": self.page_columns,
            "page_rows": self.page_rows,
            "round_number": self.round_number,
            "font_size": self.font_size,
            "round_font_size": self.round_font_size,
            "event_title_font_size": self.event_title_font_size,
        }
        positive_floats = {
            "card_width": self.card_width,
            "card_height": self.card_height,
        }
        non_negative_floats = {
            "page_margin_top": self.page_margin_top,
            "page_margin_bottom": self.page_margin_bottom,
            "page_margin_left": self.page_margin_left,
            "page_margin_right": self.page_margin_right,
            "card_gap": self.card_gap,
            "header_gap": self.header_gap,
            "border_width": self.border_width,
        }

        for key, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"'{key}' must be > 0. Got {value}.")
        for key, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"'{key}' must be > 0. Got {value}.")
        for key, value in non_negative_floats.items():
            if value < 0:
                raise ValueError(f"'{key}' must be >= 0. Got {value}.")

    def to_dict(self) -> dict:
        """Return a serializable snapshot of effective configuration."""
        return {
            "page_columns": self.page_columns,
            "page_rows": self.page_rows,
            "cards_per_page": self.cards_per_page,
            "pages": self.pages,
            "page_size": self.page_size,
            "page_margin_top": self.page_margin_top,
            "page_margin_bottom": self.page_margin_bottom,
            "page_margin_left": self.page_margin_left,
            "page_margin_right": self.page_margin_right,
            "card_gap": self.card_gap,
            "round_number": self.round_number,
            "card_width": self.card_width,
            "card_height": self.card_height,
            "header_gap": self.header_gap,
            "event_title": self.event_title,
            "round_font_size": self.round_font_size,
            "event_title_font_size": self.event_title_font_size,
            "font_size": self.font_size,
            "number_bold": self.number_bold,
            "empty_image_path": self.empty_image_path,
            "number_color": self.number_color,
            "empty_color": self.empty_color,
            "header_text_color": self.header_text_color,
            "border_color": self.border_color,
            "border_width": self.border_width,
        }

class BingoCard:
    """Represents a single 90-ball bingo card"""
    
    def __init__(self):
        self.grid = [[None for _ in range(9)] for _ in range(3)]
        self.numbers_used = set()
        
    def generate(self) -> bool:
        """Generate a valid 90-ball bingo card"""
        # Column ranges for 90-ball bingo
        column_ranges = [
            (1, 9),   # Column 0: 1-9
            (10, 19), # Column 1: 10-19
            (20, 29), # Column 2: 20-29
            (30, 39), # Column 3: 30-39
            (40, 49), # Column 4: 40-49
            (50, 59), # Column 5: 50-59
            (60, 69), # Column 6: 60-69
            (70, 79), # Column 7: 70-79
            (80, 90)  # Column 8: 80-90
        ]
        
        # Each row must have exactly 5 numbers and 4 empty spaces
        # Each column can have 0, 1, 2, or 3 numbers
        
        # First, determine how many numbers each column will have
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            attempts += 1
            self.grid = [[None for _ in range(9)] for _ in range(3)]
            self.numbers_used = set()
            
            # Generate a row-based layout: each row has exactly 5 numbers
            row_layout = self._generate_row_layout()
            
            # Fill each column according to the rows chosen in the layout
            success = True
            for col in range(9):
                target_rows = [row for row in range(3) if row_layout[row][col]]
                if not self._fill_column_with_rows(col, target_rows, column_ranges[col]):
                    success = False
                    break
                    
            if success and self._validate_card():
                return True
                
        return False
    
    def _generate_row_layout(self) -> List[List[bool]]:
        """Generate a row-based layout: 3 rows x 9 columns with 5 numbers per row.
        Returns a boolean grid indicating where numbers should be placed.
        Every column has at least one number (90-ball bingo rule).
        """
        layout = [[False for _ in range(9)] for _ in range(3)]

        # Place one number per column, spread across rows (each row gets <= 5)
        cols = list(range(9))
        random.shuffle(cols)
        row_counts = self._initial_row_counts()
        col_idx = 0
        for row, count in enumerate(row_counts):
            for _ in range(count):
                layout[row][cols[col_idx]] = True
                col_idx += 1

        # Fill each row to exactly 5 numbers
        for row in range(3):
            need = 5 - sum(layout[row])
            available = [col for col in range(9) if not layout[row][col]]
            for col in random.sample(available, need):
                layout[row][col] = True

        return layout

    def _initial_row_counts(self) -> List[int]:
        """Random row counts that sum to 9 (one per column), each at most 5."""
        while True:
            counts = [random.randint(0, 5) for _ in range(3)]
            if sum(counts) == 9:
                return counts

    def _fill_column_with_rows(self, col: int, target_rows: List[int], number_range: Tuple[int, int]) -> bool:
        """Fill a column placing numbers specifically in the given target rows."""
        count = len(target_rows)
        if count == 0:
            return True
        
        start, end = number_range
        available_numbers = [n for n in range(start, end + 1) if n not in self.numbers_used]
        
        if len(available_numbers) < count:
            return False
        
        selected_numbers = random.sample(available_numbers, count)
        selected_numbers.sort()
        
        # Place numbers in specified rows. Sort rows to keep a consistent top-to-bottom order.
        for idx, row in enumerate(sorted(target_rows)):
            self.grid[row][col] = selected_numbers[idx]
            self.numbers_used.add(selected_numbers[idx])
        
        return True

    def _validate_card(self) -> bool:
        """Validate that the card meets 90-ball bingo rules"""
        # Check that each row has exactly 5 numbers
        for row in range(3):
            numbers_in_row = sum(1 for cell in self.grid[row] if cell is not None)
            if numbers_in_row != 5:
                return False
        
        # Check that we have exactly 15 numbers total
        total_numbers = sum(1 for row in self.grid for cell in row if cell is not None)
        if total_numbers != 15:
            return False
            
        # Check that numbers are in correct columns
        column_ranges = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 49), 
                        (50, 59), (60, 69), (70, 79), (80, 90)]
        
        for col in range(9):
            start, end = column_ranges[col]
            numbers_in_col = 0
            for row in range(3):
                num = self.grid[row][col]
                if num is not None:
                    numbers_in_col += 1
                    if num < start or num > end:
                        return False
            if numbers_in_col == 0:
                return False
        
        return True
    
    def to_tuple(self) -> Tuple:
        """Convert card to tuple for hashing and comparison"""
        return tuple(tuple(row) for row in self.grid)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, BingoCard):
            return False
        return self.to_tuple() == other.to_tuple()

class BingoCardGenerator:
    """Main class for generating bingo cards and PDFs"""
    
    def __init__(self, config: BingoConfig):
        self.config = config
        self.generated_cards: Set[BingoCard] = set()
        self._empty_cell_image_cache: dict = {}
    
    def generate_unique_cards(self, num_cards: int) -> List[BingoCard]:
        """Generate a specified number of unique bingo cards"""
        cards = []
        attempts = 0
        max_attempts = num_cards * 100  # Reasonable limit
        
        while len(cards) < num_cards and attempts < max_attempts:
            attempts += 1
            card = BingoCard()
            
            if card.generate() and card not in self.generated_cards:
                cards.append(card)
                self.generated_cards.add(card)
        
        if len(cards) < num_cards:
            print(f"Warning: Could only generate {len(cards)} unique cards out of {num_cards} requested")
        
        return cards
    
    def create_pdf(self, cards: List[BingoCard], filename: str = "bingo_cards.pdf"):
        """Create a PDF with the generated bingo cards"""
        round_style, event_title_style = self._build_header_styles()
        self._validate_layout_fit(round_style, event_title_style)

        doc = BaseDocTemplate(
            filename,
            pagesize=self.config.page_size,
            leftMargin=self.config.page_margin_left * mm,
            rightMargin=self.config.page_margin_right * mm,
            topMargin=self.config.page_margin_top * mm,
            bottomMargin=self.config.page_margin_bottom * mm,
        )
        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id='normal',
        )
        doc.addPageTemplates([PageTemplate(id='normal', frames=[frame])])
        
        story = []
        
        # Process cards in pages
        for page_start in range(0, len(cards), self.config.cards_per_page):
            page_cards = cards[page_start:page_start + self.config.cards_per_page]

            # Create fixed cards grid for this page (2 columns x 4 rows by default)
            cards_table = self._create_cards_grid(page_cards, round_style, event_title_style)
            story.append(cards_table)
            
            # Add page break if not last page
            if page_start + self.config.cards_per_page < len(cards):
                story.append(PageBreak())
        
        doc.build(story)
        cell_w_mm, cell_h_mm = self.get_cell_size_mm()
        print(f"Bingo cell size: {cell_w_mm:.2f}mm x {cell_h_mm:.2f}mm")
        print(f"PDF created: {filename}")

    def _build_header_styles(self) -> Tuple[ParagraphStyle, ParagraphStyle]:
        """Create header styles used by both rendering and layout validation."""
        styles = getSampleStyleSheet()
        round_style = ParagraphStyle(
            'RoundTitle',
            parent=styles['Heading5'],
            fontSize=self.config.round_font_size,
            alignment=0,  # Left alignment
            textColor=colors.HexColor(self.config.header_text_color),
            leading=self.config.round_font_size * 1.2,
            spaceAfter=0
        )
        event_title_style = ParagraphStyle(
            'EventTitleRight',
            parent=styles['Heading5'],
            fontSize=self.config.event_title_font_size,
            alignment=2,  # Right alignment
            textColor=colors.HexColor(self.config.header_text_color),
            leading=self.config.event_title_font_size * 1.2,
            spaceBefore=0,
            spaceAfter=0
        )
        return round_style, event_title_style

    def _get_number_font_name(self) -> str:
        """Return the ReportLab font name for bingo numbers."""
        return 'Helvetica-Bold' if self.config.number_bold else 'Helvetica'

    def _get_number_leading(self) -> float:
        """Leading that vertically centers digit glyphs (no descenders) in a Table cell.

        ReportLab's plain-string cell renderer positions the baseline at
        (rowheight + leading) / 2 - fontsize. Setting leading == fontsize (the naive
        choice) leaves a residual offset proportional to fontsize, so it only looks
        centered at small sizes. Solving for the leading that puts the glyph's visual
        midpoint (ascent / 2 above baseline) at the cell's center gives this formula.
        """
        ascent, _descent = getAscentDescent(self._get_number_font_name(), self.config.font_size)
        return 2 * self.config.font_size - ascent

    def get_cell_size_mm(self) -> Tuple[float, float]:
        """Return bingo table cell size in millimeters."""
        round_style, event_title_style = self._build_header_styles()
        table_height_pt = self._get_bingo_table_height_pt(round_style, event_title_style)
        table_height_mm = table_height_pt / mm
        return (self.config.card_width / 9.0, table_height_mm / 3.0)

    def _create_cards_grid(
        self,
        cards: List[BingoCard],
        round_style: ParagraphStyle,
        event_title_style: ParagraphStyle
    ) -> Table:
        """Create a fixed grid for cards with configurable rows and columns."""
        grid_data: List[List[object]] = []
        card_index = 0

        total_columns = self.config.page_columns * 2 - 1
        for row_idx in range(self.config.page_rows):
            row_flowables: List[object] = []
            for col_idx in range(total_columns):
                if col_idx % 2 == 1:
                    row_flowables.append(Spacer(self.config.card_gap * mm, 1))
                    continue

                if card_index < len(cards):
                    row_flowables.append(
                        self._create_card_container(cards[card_index], round_style, event_title_style)
                    )
                    card_index += 1
                else:
                    row_flowables.append(Spacer(1, self.config.card_height * mm))
            grid_data.append(row_flowables)
            if row_idx < self.config.page_rows - 1:
                grid_data.append([Spacer(1, self.config.card_gap * mm) for _ in range(total_columns)])

        col_widths = []
        for col_idx in range(total_columns):
            if col_idx % 2 == 0:
                col_widths.append(self.config.card_width * mm)
            else:
                col_widths.append(self.config.card_gap * mm)

        row_heights = []
        for row_idx in range(len(grid_data)):
            if row_idx % 2 == 0:
                row_heights.append(None)
            else:
                row_heights.append(self.config.card_gap * mm)

        grid_table = Table(
            grid_data,
            colWidths=col_widths,
            rowHeights=row_heights,
            splitByRow=0,
        )
        # Keep grid anchored to the left so horizontal margins are predictable.
        grid_table.hAlign = 'LEFT'
        grid_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('NOSPLIT', (0, 0), (-1, -1)),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        return grid_table

    def _create_card_container(
        self,
        card: BingoCard,
        round_style: ParagraphStyle,
        event_title_style: ParagraphStyle
    ) -> Table:
        """Create a card container with a one-line header and the bingo table."""
        header_row = self._build_card_header_row(round_style, event_title_style)
        header_height = self._get_header_row_height_pt(round_style, event_title_style)
        header_gap_pt = self.config.header_gap * mm
        table_height_pt = self._get_bingo_table_height_pt(round_style, event_title_style)
        card_table = self._create_single_card_table(card, table_height_pt)
        container = Table(
            [[header_row], [Spacer(1, self.config.header_gap * mm)], [card_table]],
            colWidths=[self.config.card_width * mm],
            rowHeights=[header_height, header_gap_pt, table_height_pt],
        )
        container.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        return container

    def _get_header_column_widths(
        self,
        round_style: ParagraphStyle,
    ) -> Tuple[float, float]:
        """Size columns from content so the title gets all space not used by RONDA."""
        card_width_pt = self.config.card_width * mm
        header_gap_pt = 1 * mm
        round_label = f"RONDA {self.config.round_number}"
        round_width_pt = stringWidth(round_label, round_style.fontName, round_style.fontSize)
        left_width = round_width_pt + header_gap_pt
        return left_width, card_width_pt - left_width

    def _build_card_header_row(
        self,
        round_style: ParagraphStyle,
        event_title_style: ParagraphStyle
    ) -> Table:
        round_text = Paragraph(f"RONDA {self.config.round_number}", round_style)
        event_title = Paragraph(self.config.event_title, event_title_style)
        left_width, right_width = self._get_header_column_widths(round_style)
        header_row = Table(
            [[round_text, event_title]],
            colWidths=[left_width, right_width],
        )
        header_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0.5 * mm),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0.5 * mm),
        ]))
        return header_row

    def _get_header_row_height_pt(
        self,
        round_style: ParagraphStyle,
        event_title_style: ParagraphStyle
    ) -> float:
        header_row = self._build_card_header_row(round_style, event_title_style)
        _, header_height = header_row.wrap(self.config.card_width * mm, 1_000_000)
        return header_height

    def _get_bingo_table_height_pt(
        self,
        round_style: ParagraphStyle,
        event_title_style: ParagraphStyle
    ) -> float:
        total_card_height_pt = self.config.card_height * mm
        header_height_pt = self._get_header_row_height_pt(round_style, event_title_style)
        table_height_pt = total_card_height_pt - header_height_pt - (self.config.header_gap * mm)
        if table_height_pt <= 0:
            raise ValueError(
                "card_height is too small for current header fonts/gap. "
                "Increase card_height or reduce round/event font sizes/header_gap."
            )
        return table_height_pt

    def _get_empty_cell_image_bytes(self, cell_width: float, cell_height: float) -> Optional[bytes]:
        """Crop and resize empty-cell image to fill the cell, preserving aspect ratio."""
        if not self.config.empty_image_path or not os.path.exists(self.config.empty_image_path):
            return None

        cache_key = (
            self.config.empty_image_path,
            int(cell_width),
            int(cell_height),
            os.path.getmtime(self.config.empty_image_path),
        )
        if cache_key in self._empty_cell_image_cache:
            return self._empty_cell_image_cache[cache_key]

        target_w = max(1, int(cell_width * 2))
        target_h = max(1, int(cell_height * 2))

        with Image.open(self.config.empty_image_path) as source:
            image = source.convert("RGBA")
            src_ratio = image.width / image.height
            target_ratio = target_w / target_h

            if src_ratio > target_ratio:
                crop_w = int(image.height * target_ratio)
                crop_h = image.height
            else:
                crop_w = image.width
                crop_h = int(image.width / target_ratio)

            left = max(0, (image.width - crop_w) // 2)
            top = max(0, (image.height - crop_h) // 2)
            right = left + crop_w
            bottom = top + crop_h

            cropped = image.crop((left, top, right, bottom))
            resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        self._empty_cell_image_cache[cache_key] = image_bytes
        return image_bytes

    def _validate_layout_fit(
        self,
        round_style: ParagraphStyle,
        event_title_style: ParagraphStyle
    ) -> None:
        """Ensure configured layout fits printable area using actual wrapped sizes."""
        page_width, page_height = self.config.page_size
        usable_width = page_width - ((self.config.page_margin_left + self.config.page_margin_right) * mm)
        usable_height = page_height - ((self.config.page_margin_top + self.config.page_margin_bottom) * mm)
        dummy_cards = [BingoCard() for _ in range(self.config.cards_per_page)]
        grid_table = self._create_cards_grid(dummy_cards, round_style, event_title_style)
        required_width, required_height = grid_table.wrap(usable_width, usable_height)

        if required_width > usable_width or required_height > usable_height:
            raise ValueError(
                "Layout does not fit page. Reduce card dimensions/rows/columns/gap or margins, "
                "or increase page size."
            )

    def _create_single_card_table(self, card: BingoCard, table_height_pt: float) -> Table:
        """Create a table for a single bingo card"""
        table_data = []
        cell_w_pt = (self.config.card_width * mm) / 9.0
        cell_h_pt = table_height_pt / 3.0
        empty_cell_image = self._get_empty_cell_image_bytes(cell_w_pt, cell_h_pt)
        
        for row in card.grid:
            row_data = []
            for cell in row:
                if cell is None:
                    # Empty cell
                    if empty_cell_image:
                        try:
                            img = RLImage(io.BytesIO(empty_cell_image), width=cell_w_pt, height=cell_h_pt)
                            row_data.append(img)
                        except Exception:
                            row_data.append("")
                    else:
                        row_data.append("")
                else:
                    row_data.append(str(cell))
            table_data.append(row_data)
        
        # Create table
        table = Table(table_data, 
                     colWidths=[cell_w_pt] * 9,
                     rowHeights=[cell_h_pt] * 3)
        
        # Style the table
        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), self.config.font_size),
            ('FONTNAME', (0, 0), (-1, -1), self._get_number_font_name()),
            ('LEADING', (0, 0), (-1, -1), self._get_number_leading()),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('GRID', (0, 0), (-1, -1), self.config.border_width, colors.HexColor(self.config.border_color)),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(self.config.number_color)),
        ]))
        
        # Style empty cells
        if not empty_cell_image:
            for row_idx, row in enumerate(card.grid):
                for col_idx, cell in enumerate(row):
                    if cell is None:
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), 
                             colors.HexColor(self.config.empty_color)),
                        ]))
        
        return table

class ProbabilityCalculator:
    """Calculate probabilities related to bingo card generation"""
    
    @staticmethod
    def calculate_total_possible_cards() -> int:
        """Calculate theoretical maximum number of unique 90-ball bingo cards"""
        # This is a complex combinatorial problem
        # Approximation: each column has C(numbers_in_range, max_numbers_in_column) possibilities
        # But we need to account for the constraint that each row has exactly 5 numbers
        
        # Simplified calculation - actual number is much higher due to complex constraints
        column_ranges = [9, 10, 10, 10, 10, 10, 10, 10, 11]  # Numbers available per column
        
        # Very rough approximation
        total = 1
        for i, range_size in enumerate(column_ranges):
            # Each column can have 0-3 numbers
            possibilities = sum(math.comb(range_size, k) for k in range(4))
            total *= possibilities
        
        # This is a vast overestimate as it doesn't account for row constraints
        print("Note: This is a rough overestimate. The actual calculation is extremely complex.")
        return total
    
    @staticmethod
    def calculate_duplicate_probability(num_cards: int) -> float:
        """Calculate probability of generating duplicate cards"""
        # Using birthday paradox approximation
        total_possible = ProbabilityCalculator.calculate_total_possible_cards()
        
        if num_cards >= total_possible:
            return 1.0
        
        # Probability of no duplicates
        prob_no_duplicates = 1.0
        for i in range(num_cards):
            prob_no_duplicates *= (total_possible - i) / total_possible
        
        return 1.0 - prob_no_duplicates
    
    @staticmethod
    def analyze_generated_cards(cards: List[BingoCard]) -> dict:
        """Analyze a set of generated cards for patterns and statistics"""
        if not cards:
            return {}
        
        analysis = {
            'total_cards': len(cards),
            'unique_cards': len(set(cards)),
            'duplicate_rate': (len(cards) - len(set(cards))) / len(cards),
            'number_frequency': {},
            'column_usage': [0] * 9,
            'row_patterns': {}
        }
        
        # Analyze number frequency
        for card in cards:
            for row in card.grid:
                for cell in row:
                    if cell is not None:
                        analysis['number_frequency'][cell] = analysis['number_frequency'].get(cell, 0) + 1
        
        # Analyze column usage
        for card in cards:
            for col in range(9):
                numbers_in_col = sum(1 for row in range(3) if card.grid[row][col] is not None)
                analysis['column_usage'][col] += numbers_in_col
        
        return analysis

# Example usage and testing
def main():
    """Example usage of the bingo card generator"""
    
    # Edit BingoConfig defaults above, or override only what you need here.
    config = BingoConfig()
    print("Active configuration:")
    print(pformat(config.to_dict(), sort_dicts=False))
    
    # Generate cards
    generator = BingoCardGenerator(config)
    num_cards = config.cards_per_page * config.pages
    cards = generator.generate_unique_cards(num_cards)
    
    print(f"Generated {len(cards)} unique cards")
    
    # Create PDF
    generator.create_pdf(cards, "my_bingo_cards.pdf")
    cell_w_mm, cell_h_mm = generator.get_cell_size_mm()
    print(f"Configured cell size: {cell_w_mm:.2f}mm x {cell_h_mm:.2f}mm")
    
    # Calculate probabilities
    calc = ProbabilityCalculator()
    
    print(f"\n--- Probability Analysis ---")
    print(f"Requested cards: {num_cards}")
    print(f"Successfully generated: {len(cards)}")
    print(f"Estimated total possible cards: {calc.calculate_total_possible_cards():,}")
    print(f"Probability of duplicates with {num_cards} cards: {calc.calculate_duplicate_probability(num_cards):.10f}")
    
    # Analyze generated cards
    analysis = calc.analyze_generated_cards(cards)
    print(f"\n--- Card Analysis ---")
    print(f"Unique cards: {analysis['unique_cards']} / {analysis['total_cards']}")
    print(f"Actual duplicate rate: {analysis['duplicate_rate']:.4f}")
    print(f"Most common numbers: {sorted(analysis['number_frequency'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Column usage distribution: {analysis['column_usage']}")

if __name__ == "__main__":
    main()