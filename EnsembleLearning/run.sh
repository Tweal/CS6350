#!/bin/bash

pip install pandas
pip install tqdm
pip install matplotlib
pip install numpy

python driver.py ada

python driver.py bagged

python driver.py bvdbagged

python driver.py rf

python driver.py bvdrf