#!/bin/bash

pip install pandas
pip install tqdm
pip install matplotlib
pip install numpy

echo "Running Neural Network with randomized initial weights."
echo "This will take a moment"
python driver.py random

echo "Running Neural Network with zero initial weights."
echo "This will take a moment"
python driver.py zero

read