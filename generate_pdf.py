#!/usr/bin/env python3
"""
90-Ball Bingo Card Generator
A highly customizable bingo card generator with PDF output and probability calculations.
"""

import random
import math
import itertools
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.platypus.flowables import KeepTogether
import io
import os

@dataclass
class BingoConfig:
    """Configuration class for bingo card generation"""
    cards_per_page: int = 4
    page_title: str = "BINGO CARDS"
    card_width: float = 80  # mm
    card_height: float = 50  # mm
    card_margin: float = 10  # mm
    cell_padding: float = 2  # mm
    font_size: int = 12
    title_font_size: int = 16
    empty_image_path: Optional[str] = None
    page_size = A4
    number_color: str = '#000000'
    empty_color: str = '#E0E0E0'
    border_color: str = '#000000'
    border_width: float = 1

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
            for col in range(8):
                target_rows = [row for row in range(3) if row_layout[row][col]]
                if not self._fill_column_with_rows(col, target_rows, column_ranges[col]):
                    success = False
                    break
                    
            if success and self._validate_card():
                return True
                
        return False
    
    def _generate_column_distribution(self) -> List[int]:
        """Generate a valid distribution of numbers across columns"""
        # Each column can have 0-3 numbers, total must be 15
        # Generate random distribution
        distribution = []
        remaining = 15
        
        for i in range(8):  # First 8 columns
            max_for_this_col = min(3, remaining - (8 - i))  # Ensure we can fill remaining columns
            min_for_this_col = max(0, remaining - 3 * (8 - i))  # Ensure we don't exceed 3 per remaining column
            
            if max_for_this_col >= min_for_this_col:
                count = random.randint(min_for_this_col, max_for_this_col)
            else:
                count = 0
            
            distribution.append(count)
            remaining -= count
        
        # Last column gets whatever remains
        distribution.append(remaining)
        
        # Validate distribution
        if any(x > 3 or x < 0 for x in distribution):
            return [0] * 9  # Invalid distribution
            
        return distribution
    
    def _generate_row_layout(self) -> List[List[bool]]:
        """Generate a row-based layout: 3 rows x 8 columns with 5 numbers per row.
        Returns a boolean grid indicating where numbers should be placed.
        """
        layout = [[False for _ in range(8)] for _ in range(3)]
        
        for row in range(3):
            chosen_cols = random.sample(range(8), 5)
            for col in chosen_cols:
                layout[row][col] = True
        
        return layout

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

    def _fill_column(self, col: int, count: int, number_range: Tuple[int, int]) -> bool:
        """Fill a column with the specified number of random numbers"""
        if count == 0:
            return True
            
        # Get available numbers for this column
        start, end = number_range
        available_numbers = [n for n in range(start, end + 1) if n not in self.numbers_used]
        
        if len(available_numbers) < count:
            return False
            
        # Select random numbers
        selected_numbers = random.sample(available_numbers, count)
        selected_numbers.sort()
        
        # Select random positions in this column for the numbers
        available_positions = [(0, col), (1, col), (2, col)]
        selected_positions = random.sample(available_positions, count)
        
        # Place numbers in selected positions
        for i, pos in enumerate(selected_positions):
            row, column = pos
            self.grid[row][column] = selected_numbers[i]
            self.numbers_used.add(selected_numbers[i])
        
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
            for row in range(3):
                num = self.grid[row][col]
                if num is not None and (num < start or num > end):
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
        doc = SimpleDocTemplate(
            filename,
            pagesize=self.config.page_size,
            leftMargin=self.config.card_margin * mm,
            rightMargin=self.config.card_margin * mm,
            topMargin=self.config.card_margin * mm,
            bottomMargin=self.config.card_margin * mm
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Create title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=self.config.title_font_size,
            alignment=1  # Center alignment
        )
        
        # Process cards in pages
        for page_start in range(0, len(cards), self.config.cards_per_page):
            page_cards = cards[page_start:page_start + self.config.cards_per_page]
            
            # Add title
            if self.config.page_title:
                story.append(Paragraph(self.config.page_title, title_style))
                story.append(Spacer(1, 10 * mm))
            
            # Create cards table for this page
            cards_table = self._create_cards_table(page_cards)
            story.append(KeepTogether(cards_table))
            
            # Add page break if not last page
            if page_start + self.config.cards_per_page < len(cards):
                story.append(Spacer(1, 20 * mm))
        
        doc.build(story)
        print(f"PDF created: {filename}")
    
    def _create_cards_table(self, cards: List[BingoCard]) -> Table:
        """Create a table layout for multiple bingo cards"""
        # Calculate cards per row based on page width
        page_width = self.config.page_size[0] - (2 * self.config.card_margin * mm)
        card_width_points = self.config.card_width * mm
        cards_per_row = max(1, int(page_width / (card_width_points + self.config.card_margin * mm)))
        
        table_data = []
        
        # Arrange cards in rows
        for i in range(0, len(cards), cards_per_row):
            row_cards = cards[i:i + cards_per_row]
            card_tables = [self._create_single_card_table(card) for card in row_cards]
            table_data.append(card_tables)
        
        # Create main table
        main_table = Table(table_data)
        main_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        return main_table
    
    def _create_single_card_table(self, card: BingoCard) -> Table:
        """Create a table for a single bingo card"""
        table_data = []
        
        for row in card.grid:
            row_data = []
            for cell in row:
                if cell is None:
                    # Empty cell
                    if self.config.empty_image_path and os.path.exists(self.config.empty_image_path):
                        try:
                            img = RLImage(self.config.empty_image_path, 
                                        width=15*mm, height=10*mm)
                            row_data.append(img)
                        except:
                            row_data.append("")
                    else:
                        row_data.append("")
                else:
                    row_data.append(str(cell))
            table_data.append(row_data)
        
        # Create table
        table = Table(table_data, 
                     colWidths=[self.config.card_width * mm / 9] * 9,
                     rowHeights=[self.config.card_height * mm / 3] * 3)
        
        # Style the table
        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), self.config.font_size),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), self.config.border_width, colors.HexColor(self.config.border_color)),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(self.config.number_color)),
        ]))
        
        # Style empty cells
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
    
    # Create configuration
    config = BingoConfig(
        cards_per_page=4,
        page_title="90-BALL BINGO CARDS",
        card_width=85,
        card_height=55,
        card_margin=15,
        font_size=10,
        title_font_size=18,
        # empty_image_path="star.png",  # Uncomment if you have an image
        number_color='#2E3440',
        empty_color='#ECEFF4',
        border_color='#4C566A'
    )
    
    # Generate cards
    generator = BingoCardGenerator(config)
    num_cards = 20
    cards = generator.generate_unique_cards(num_cards)
    
    print(f"Generated {len(cards)} unique cards")
    
    # Create PDF
    generator.create_pdf(cards, "my_bingo_cards.pdf")
    
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