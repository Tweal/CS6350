#!/bin/bash

pip install pandas
pip install tqdm
pip install matplotlib
pip install numpy
pip install scipy

clear

echo "Running Primal: This should be quick"

python driver.py primal

echo "Running Dual: This will take a moment"

python driver.py dual

echo "Running Gaussian Kernel: This will take a moment"

python driver.py gk

read