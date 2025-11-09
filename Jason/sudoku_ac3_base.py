#!/usr/bin/env python3
"""
sudoku_ac3_base.py
Base classes and data structures for AC-3 Sudoku solver

FIXED: Coordinate system now matches original implementation
- Rows: "123456789"
- Cols: "ABCDEFGHI"
- Cells: A1, A2, ..., I9
"""

from typing import List, Dict, Set, Tuple
import copy

# Grid coordinates - CORRECTED to match original
ROWS = "123456789"
COLS = "ABCDEFGHI"


class SudokuAC3:
    """Sudoku representation with constraint propagation support."""
    
    def __init__(self, grid: str):
        """
        Initialize Sudoku from 81-character string.
        '0' represents empty cells.
        """
        if len(grid) != 81:
            raise ValueError(f"Grid must be 81 characters, got {len(grid)}")
        
        # Generate all cell coordinates (A1, A2, ..., I9)
        self.cells = self._generate_coords()
        
        # Generate possibilities for each cell
        self.possibilities = self._generate_possibilities(grid)
        
        # Generate constraints
        rule_constraints = self._generate_rules_constraints()
        self.binary_constraints = self._generate_binary_constraints(rule_constraints)
        
        # Generate related cells for each cell
        self.related_cells = self._generate_related_cells()
        
        # Pruned values tracker (for backtracking)
        self.pruned = {cell: [] for cell in self.cells}
    
    def _generate_coords(self) -> List[str]:
        """Generate all cell coordinates (A1, A2, ..., I9)."""
        all_cells = []
        for col in COLS:  # A, B, C, ..., I
            for row in ROWS:  # 1, 2, 3, ..., 9
                all_cells.append(col + row)
        return all_cells
    
    def _generate_possibilities(self, grid: str) -> Dict[str, List[int]]:
        """Generate possible values for each cell."""
        possibilities = {}
        grid_list = list(grid)
        
        for index, coords in enumerate(self.cells):
            if grid_list[index] == '0':
                # Empty cell - all values possible
                possibilities[coords] = list(range(1, 10))
            else:
                # Fixed value
                possibilities[coords] = [int(grid_list[index])]
        
        return possibilities
    
    def _generate_rules_constraints(self) -> List[List[str]]:
        """Generate row, column, and box constraints."""
        row_constraints = []
        column_constraints = []
        square_constraints = []
        
        # Row constraints (all cells in same row)
        for row in ROWS:
            row_constraints.append([col + row for col in COLS])
        
        # Column constraints (all cells in same column)
        for col in COLS:
            column_constraints.append([col + row for row in ROWS])
        
        # Box constraints (3x3 squares)
        # Split into groups of 3
        cols_groups = [COLS[i:i+3] for i in range(0, len(COLS), 3)]
        rows_groups = [ROWS[i:i+3] for i in range(0, len(ROWS), 3)]
        
        for col_group in cols_groups:
            for row_group in rows_groups:
                square = []
                for col in col_group:
                    for row in row_group:
                        square.append(col + row)
                square_constraints.append(square)
        
        return row_constraints + column_constraints + square_constraints
    
    def _generate_binary_constraints(self, rule_constraints: List[List[str]]) -> List[Tuple[str, str]]:
        """Convert rule constraints to binary constraints."""
        binary_constraints = []
        seen = set()
        
        for constraint_set in rule_constraints:
            # Generate all directed pairs
            for i in range(len(constraint_set)):
                for j in range(len(constraint_set)):
                    if i != j:
                        pair = (constraint_set[i], constraint_set[j])
                        if pair not in seen:
                            binary_constraints.append(pair)
                            seen.add(pair)
        
        return binary_constraints
    
    def _generate_related_cells(self) -> Dict[str, List[str]]:
        """Generate list of related cells for each cell."""
        related_cells = {}
        
        for cell in self.cells:
            related_cells[cell] = []
            
            # Find all cells that have constraints with current cell
            for constraint in self.binary_constraints:
                if cell == constraint[0]:
                    related_cells[cell].append(constraint[1])
        
        return related_cells
    
    def is_finished(self) -> bool:
        """Check if Sudoku is completely solved."""
        for coords, possibilities in self.possibilities.items():
            if len(possibilities) > 1:
                return False
        return True
    
    def is_valid(self) -> bool:
        """Check if current state has no empty domains."""
        for possibilities in self.possibilities.values():
            if len(possibilities) == 0:
                return False
        return True
    
    def get_solution_string(self) -> str:
        """Convert solved Sudoku back to 81-character string."""
        result = []
        for cell in self.cells:
            if len(self.possibilities[cell]) == 1:
                result.append(str(self.possibilities[cell][0]))
            else:
                result.append('0')
        return ''.join(result)
    
    def print_grid(self) -> None:
        """Print the grid in formatted style."""
        print()
        # Need to print in row-major order for readability
        for row in ROWS:
            row_cells = []
            for col in COLS:
                cell = col + row
                if len(self.possibilities[cell]) == 1:
                    val = str(self.possibilities[cell][0])
                else:
                    val = '.'
                row_cells.append(val)
            
            # Print with box separators
            print(" ".join(row_cells[0:3]) + " | " + 
                  " ".join(row_cells[3:6]) + " | " + 
                  " ".join(row_cells[6:9]))
            
            if row in ['3', '6']:
                print("-" * 21)
        print()