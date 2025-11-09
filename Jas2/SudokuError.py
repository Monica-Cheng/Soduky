# Original repository: https://github.com/paccionesawyer/sudokuSolver-CSP
# License: MIT
"""
SudokuError
Purpose:    Throw exceptions when there is an error input
Parameters: Exception [string], The text to be printed when the exception 
            is thrown
Returns:    Nothing
Effects:    If this class is called the program stops and prints 'Exception'
Notes:      Parent class is Exception
"""
class SudokuError(Exception):
    pass