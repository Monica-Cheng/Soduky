#!/usr/bin/env python3
"""
sudoku_ac3_backtrack.py
Backtracking algorithm with forward checking

FIXED: Proper domain copying to avoid reference issues
"""

from typing import Dict, Optional
import copy
from sudoku_ac3_base import SudokuAC3
from sudoku_ac3_heuristics import select_unassigned_variable, order_domain_values


def is_consistent(sudoku: SudokuAC3, assignment: Dict[str, int], 
                  cell: str, value: int) -> bool:
    """
    Check if assigning value to cell is consistent with current assignment.
    
    Returns True if no conflicts, False otherwise.
    """
    for current_cell, current_value in assignment.items():
        # If same value and cells are related, inconsistent
        if current_value == value and current_cell in sudoku.related_cells[cell]:
            return False
    
    return True


def forward_check(sudoku: SudokuAC3, cell: str, value: int, assignment: Dict[str, int]) -> None:
    """
    Forward checking: Remove value from domains of related unassigned cells.
    Records removed values in pruned for backtracking.
    """
    for related_cell in sudoku.related_cells[cell]:
        # If unassigned and value is in domain
        if related_cell not in assignment:
            if value in sudoku.possibilities[related_cell]:
                # Remove from domain
                sudoku.possibilities[related_cell].remove(value)
                # Record for backtracking
                sudoku.pruned[cell].append((related_cell, value))


def assign(sudoku: SudokuAC3, cell: str, value: int, assignment: Dict[str, int]) -> None:
    """
    Assign value to cell and perform forward checking.
    """
    assignment[cell] = value
    
    # Perform forward checking
    forward_check(sudoku, cell, value, assignment)


def unassign(sudoku: SudokuAC3, cell: str, assignment: Dict[str, int]) -> None:
    """
    Unassign cell (backtrack) and restore pruned values.
    
    FIXED: Properly restore all pruned values
    """
    if cell in assignment:
        # Restore pruned values
        for coord, value in sudoku.pruned[cell]:
            # Make sure we don't add duplicates
            if value not in sudoku.possibilities[coord]:
                sudoku.possibilities[coord].append(value)
        
        # Clear pruned list
        sudoku.pruned[cell] = []
        
        # Remove assignment
        del assignment[cell]


def recursive_backtrack(assignment: Dict[str, int], sudoku: SudokuAC3, 
                       metrics: dict) -> Optional[Dict[str, int]]:
    """
    Recursive backtracking algorithm with forward checking.
    
    FIXED: Better handling of domain consistency
    
    Args:
        assignment: Current partial assignment
        sudoku: SudokuAC3 instance
        metrics: Dictionary to track nodes and backtracks
    
    Returns:
        Complete assignment if solution found, None otherwise
    """
    # If assignment is complete, return it
    if len(assignment) == len(sudoku.cells):
        return assignment
    
    # Select unassigned variable using MRV heuristic
    cell = select_unassigned_variable(assignment, sudoku)
    
    if cell is None:
        return assignment
    
    # Get ordered values using LCV heuristic
    ordered_values = order_domain_values(sudoku, cell)
    
    # Try each value
    for value in ordered_values:
        metrics['nodes'] += 1
        
        # Check if value is consistent
        if is_consistent(sudoku, assignment, cell, value):
            # Save current state before assignment
            saved_possibilities = {}
            for c in sudoku.cells:
                saved_possibilities[c] = sudoku.possibilities[c][:]
            
            # Assign value
            assign(sudoku, cell, value, assignment)
            
            # Check if any domain became empty (forward checking detected conflict)
            domain_valid = all(len(sudoku.possibilities[c]) > 0 
                             for c in sudoku.cells if c not in assignment)
            
            if domain_valid:
                # Recursive call
                result = recursive_backtrack(assignment, sudoku, metrics)
                
                # If successful, return
                if result:
                    return result
            
            # Backtrack - restore state
            unassign(sudoku, cell, assignment)
            metrics['backtracks'] += 1
            
            # Restore possibilities to saved state
            for c in sudoku.cells:
                sudoku.possibilities[c] = saved_possibilities[c]
    
    # No solution found
    return None