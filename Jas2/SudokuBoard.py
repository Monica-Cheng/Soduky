# Modified from: Sawyer Bailey Paccione, Tufts University, 2020
# Original repository: https://github.com/paccionesawyer/sudokuSolver-CSP
# License: MIT
# Modification: print divider every 3 rows / also able to read the txt file with 81 digits
from SudokuError import SudokuError
from copy import copy, deepcopy


class SudokuBoard(object):
    ############################################################################
    #                              Public Members                              #
    ############################################################################
    """
    __init__
    Purpose:    Initialize values in the sudoku puzzle
    Parameters: Nothing
    Returns:    Nothing
    Effects:    Memory, self.board
    Notes:      If no board text file is provided the sudoku puzzle solver
                will use the standard "easy_puzzle.txt"
                self.conflict_stack is a list of row, col positions for 
                backjumping.
    """
    def __init__(self, input_file):
        self.__initialize_board(input_file)
        self.unique_states      = 1
        self.conflict_stacks    = [[[] for j in range(9)] for i in range(9)]
        self.backjump           = [-1, -1]
        self.gameover           = False
        self.backtracks         = 0

    """
    start_over
    Purpose:    Returns the board to its original position
    Parameters: Nothing
    Returns:    Nothing
    Effects:    self.board
    Notes:      
    """
    def start_over(self):
        self.board = deepcopy(self.start_board)

    """
    print_board 
    Purpose:    Prints the sudoku puzzle to the terminal in a fancy manner
    Parameters: Nothing
    Returns:    Nothing
    Effects:    The Terminal
    Notes:      
    """
    def print_board(self):
        """Print Sudoku board neatly like standard layout."""
        for i in range(9):
            if i > 0 and i % 3 == 0:
                print("- - - - - - - - - - -")
            
            for j in range(9):
                print(self.board[i][j], end=" ")
                if (j + 1) % 3 == 0 and j != 8:
                    print("|", end=" ")
            print()  # new line after each full row



    """
    valid_input
    Purpose:    Check to see if most recent input maintains the boards validity 
    Parameters: row [int], the row index of the most recently inserted number
                col [int], the column index of the most recently inserted num
                num [int], the most recently number
    Returns:    [bool] The validity of the box, row, and column  of the most 
                recently inserted number
    Effects:    solver.py
    Notes:      
    """
    def valid_input(self, row, col, num):
        return (self.__valid_box(row, col, num) 
                and self.__valid_row(row, col, num) 
                and self.__valid_col(row, col, num))

    """
    combine_conflicts
    Purpose:    Combine two conflict sets
    Parameters: source  [list], Conflict set of the position with the empty or 
                                exhausted domain
                dest    [list], Conflict set of the position you are jumping
                                back to
    Returns:    Nothing
    Effects:    Empties the conflict stack of the source and puts it into the 
                source
    Notes:      Not sure whether or not to prepend or append
    """
    def combine_conflicts(self, source, dest):
        for to_add in reversed(source):
            if(to_add not in dest):
                # print("Adding", to_add, "to", dest)
                dest.insert(0, to_add)

    """
    check_win
    Purpose:    Checks to see if the current board is filled and valid
    Parameters: Nothing
    Returns:    [bool] True if it is a win, False if it is not a win
    Effects:    gameover
    Notes:
    """
    def check_win(self):
        for i in range(len((self.board))):
            for j in range(len(self.board[i])):
                if not self.valid_input(i, j, self.board[i][j]):
                    return False
        self.gameover = True
        return True


    ############################################################################
    #                             Private  Members                             #
    ############################################################################
    """
    initialize
    Purpose:    Initialize the Sudoku board from a .txt file.
    Supports both single-line (81 digits) and multi-line inputs.
    Parameters: input_file [string], the .txt file containing the Sudoku puzzle.
    Returns:    Nothing
    Effects:    Reads digits into a 9x9 grid and stores it in self.board.
    Notes:      0 represents an empty cell.
"""

    def __initialize_board(self, input):
        content = input.read().strip().replace(" ", "").replace("\n", "")
        if len(content) != 81:
            raise SudokuError("Invalid Sudoku file format: must contain 81 digits.")
        
        # Create a 9×9 grid
        self.board = [[0 for _ in range(9)] for _ in range(9)]
        idx = 0
        for i in range(9):
            for j in range(9):
                char = content[idx]
                if not char.isdigit():
                    raise SudokuError("File must contain only digits 0–9.")
                self.board[i][j] = int(char)
                idx += 1
                
        input.close()
        self.start_board = deepcopy(self.board)



    """
    __valid_table
    Purpose:    Check to see if the table is valid
    Parameters: Nothing
    Returns:    [bool], True if the table is valid, False if not
    Effects:    Nothing
    Notes:
    """
    def __valid_table(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if (self.board[i][j] != 0 
                    and not self.valid_input(i, j, self.board[i][j])):
                    print([i, j])
                    return False
        return True

    """
    valid_box
    Purpose:    check to see if the box of the most recently inserted number is
                valid. (3 x 3)
    Parameters: row [int], the row index of the most recently inserted number
                col [int], the column index of the most recently inserted num
                num [int], the most recently number
    Returns:    [bool], the validity of the (3 x 3) num was inserted in
    Effects:    check_input
    Notes:      
    """
    def __valid_box(self, row, col, num):
        box_x = col // 3
        box_y = row // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if (self.board[i][j] == num and i != row and j != col):
                    # print("Not Valid B")
                    return False
        return True

    """
    valid_row
    Purpose:    Check to see if the row of the most recently inserted number is
                valid
    Parameters: row [int], the row index of the most recently inserted number
                col [int], the column index of the most recently inserted num
                num [int], the most recently number
    Returns:    [bool], the validity of the row in the sudoku puzzle 
    Effects:    check_input
    Notes:      
    """
    def __valid_row(self, row, col, num):
        for j in range(len(self.board[row])):
            if (self.board[row][j] == num and col != j):
                # print("Not Valid r")
                return False
        return True

    """
    valid_col
    Purpose:    Check to see if the col of the most recently inserted number is
                valid
    Parameters: row [int], the row index of the most recently inserted number
                col [int], the column index of the most recently inserted num
                num [int], the most recently number
    Returns:    [bool], the validity of the col in the sudoku puzzle 
    Effects:    check_input
    Notes:      
    """
    def __valid_col(self, row, col, num):
        for i in range(len(self.board)):
            if (self.board[i][col] == num and row != i):
                # print("Not Valid C")
                return False
        return True