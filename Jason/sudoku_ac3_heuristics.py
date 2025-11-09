#!/usr/bin/env python3
"""
sudoku_ac3_heuristics.py
MRV and LCV heuristics for variable and value ordering
"""

from typing import List, Optional
from sudoku_ac3_base import SudokuAC3


def number_of_conflicts(sudoku: SudokuAC3, cell: str, value: int) -> int:
    """
    Count the number of conflicts for a cell with a specific value.
    
    A conflict occurs when a related cell has not been assigned yet
    and the value exists in its possibilities.
    """
    count = 0
    
    for related_cell in sudoku.related_cells[cell]:
        # If related cell is unassigned and value is in its domain
        if (len(sudoku.possibilities[related_cell]) > 1 and 
            value in sudoku.possibilities[related_cell]):
            count += 1
    
    return count


def select_unassigned_variable(assignment: dict, sudoku: SudokuAC3) -> Optional[str]:
    """
    MRV (Minimum Remaining Values) Heuristic.
    
    Select the unassigned variable with the fewest possible values remaining.
    This is also known as "most constrained variable" heuristic.
    
    Returns:
        The cell with minimum remaining values, or None if all assigned
    """
    unassigned = []
    
    # Collect all unassigned cells
    for cell in sudoku.cells:
        if cell not in assignment:
            unassigned.append(cell)
    
    if not unassigned:
        return None
    
    # Return cell with minimum domain size (MRV)
    return min(unassigned, key=lambda cell: len(sudoku.possibilities[cell]))


def order_domain_values(sudoku: SudokuAC3, cell: str) -> List[int]:
    """
    LCV (Least Constraining Value) Heuristic.
    
    Order the values for a cell, preferring values that rule out
    the fewest choices for neighboring variables.
    
    Returns:
        List of values sorted by least constraining first
    """
    # If only one possibility, return it
    if len(sudoku.possibilities[cell]) == 1:
        return sudoku.possibilities[cell]
    
    # Sort by number of conflicts (ascending - least conflicts first)
    return sorted(
        sudoku.possibilities[cell], 
        key=lambda value: number_of_conflicts(sudoku, cell, value)
    )