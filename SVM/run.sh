#!/bin/bash

pip install pandas
pip install tqdm
pip install matplotlib
pip install numpy
pip install scipy

python driver.py primal

python driver.py dual

python driver.py gk

read