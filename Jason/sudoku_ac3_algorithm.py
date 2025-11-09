#!/usr/bin/env python3
"""
sudoku_ac3_algorithm.py
AC-3 (Arc Consistency 3) algorithm implementation

FIXED: Proper handling of domain modifications during AC-3
"""

from typing import List, Tuple, Optional
import copy
from sudoku_ac3_base import SudokuAC3


def is_different(val1: int, val2: int) -> bool:
    """Check if two values are different."""
    return val1 != val2


def remove_inconsistent_values(sudoku: SudokuAC3, cell_i: str, cell_j: str) -> bool:
    """
    Remove values from cell_i's domain that are inconsistent with cell_j.
    Returns True if a value was removed.
    
    FIXED: Create copy of domain before iterating
    """
    removed = False
    
    # Make a copy to avoid modifying list during iteration
    values_to_check = sudoku.possibilities[cell_i][:]
    
    for value in values_to_check:
        # Check if value is consistent with any value in cell_j
        # A value is consistent if there exists a different value in cell_j's domain
        has_consistent_value = any(
            is_different(value, poss) 
            for poss in sudoku.possibilities[cell_j]
        )
        
        if not has_consistent_value:
            # Remove inconsistent value
            if value in sudoku.possibilities[cell_i]:
                sudoku.possibilities[cell_i].remove(value)
                removed = True
    
    return removed


def ac3(sudoku: SudokuAC3, queue: Optional[List[Tuple[str, str]]] = None) -> bool:
    """
    AC-3 Algorithm for constraint propagation.
    
    Implementation follows the original Python code exactly.
    
    Args:
        sudoku: SudokuAC3 instance
        queue: Optional initial queue of arcs
    
    Returns:
        True if arc consistency achieved, False if no solution exists
    """
    # Initialize queue with all binary constraints
    if queue is None:
        queue = list(sudoku.binary_constraints)
    
    while queue:
        # Get next arc
        xi, xj = queue.pop(0)
        
        # Try to remove inconsistent values
        if remove_inconsistent_values(sudoku, xi, xj):
            # If domain becomes empty, no solution exists
            if len(sudoku.possibilities[xi]) == 0:
                return False
            
            # Add all arcs (Xk, Xi) where Xk is neighbor of Xi
            # but not Xj (the arc we just processed)
            for xk in sudoku.related_cells[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    
    return True