# Soduky

# Algorithm 1: Basic Backtracking (mon)

python3 sudoku_runner.py easy.txt medium.txt hard.txt

python3 mon_updated.py easy.txt medium.txt hard.txt

# Algorithm 2: Forward Checking + Heuristics (jas)

python3 sudoku_runner_forward_checking.py easy.txt medium.txt hard.txt

python3 sudoku_runner_fc_minimal.py easy.txt medium.txt hard.txt

# Algorithm 3: AC-3 + FC + Heuristics (Jason)

python3 sudoku_runner_ac3_original.py easy.txt medium.txt hard.txt

python3 jason_updated.py easy.txt medium.txt hard.txt

# Algorithm 4: Cindy

python3 cindy_updated.py easy.txt medium.txt hard.txt

# To run cnn

python sudoku_main.py --img_fpath "data/sudoku_images/22.jpg" --model_fpath "models/model_fonts_mnist.keras" --output_path "recognized_sudoku.txt"

# To commit to git hub

git add .

git commit -m "Added Jason folder with AC3 solver and implemented .gitignore."

git push

# Activate env

source .venv/bin/activate

# Git Hub sources

mon: https://github.com/CharKeaney/sudoku-solver/blob/master/sudosolver.py

cindy:

Jason:

Jas2: https://github.com/paccionesawyer/sudokuSolver-CSP

CNN:
