# Soduky

# Algorithm 1: Basic Backtracking (mon)

python3 algo1_run.py easy.txt medium.txt hard.txt

# Algorithm 2: Backtracking + Forward checking + MRV (Jasmine)

python3 algo2_run.py easy.txt medium.txt hard.txt

# Algorithm 3: AC-3 + FC + Heuristics (Jason)

python3 algo3_run.py easy.txt medium.txt hard.txt

# Algorithm 4: MIN CONFLICT (CINDY)

python3 algo4_run.py easy.txt medium.txt hard.txt

# To run cnn

python sudoku_main.py --img_fpath "data/sudoku_images/2.jpg" --output_path "recognized_sudoku.txt"

# To commit to git hub

git add .

git commit -m "Added Jason folder with AC3 solver and implemented .gitignore."

git push

# Activate env

source .venv/bin/activate

# Git Hub sources

mon: https://github.com/CharKeaney/sudoku-solver/blob/master/sudosolver.py

cindy:

1. https://doi.org/10.1016/0004-3702(92)90007-K

2. https://aima.cs.berkeley.edu/

3. https://github.com/kushjain/Min-Conflicts

Jason: https://github.com/stressGC/Python-AC3-Backtracking-CSP-Sudoku-Solver/tree/master

Jas2: https://github.com/paccionesawyer/sudokuSolver-CSP

CNN: https://github.com/rg1990/cv-sudoku-solver

Run Combine File:
python sudoku_integrated_solver.py --image external1.jpg
