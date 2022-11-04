# Ensemble
This is a collection of ensemble methods to enhance decision tree performance.

## AdaBoost
An implementation of the AdaBoost algorithm using entropy as a purity method.

### How to Use
adaboost.py can be ran directly with `python adaboost.py` doing so will result in a small mock dataset being generated and used.

To use it externally you need to use the provided constructor for the ADABoost class. The constructor has 3 required parameters. A pandas dataframe of the data to be trained on, a map mapping attribute names to acceptable values, and a string for the label to be predicted. 

From there there are two options to run the algorithm. You can repeatedly call .run_single() and have it run a single iteration of the algorithm generating a single tree at a time. Or you can just call .run(), this method has an optional parameter T that controlls how many iterations are ran. T defaults to 1000. 

Once your trees are generated you can classify data by calling .classify() and either passing a single row or a pandas dataframe. It will return a series of labels if a dataframe was passed and a single value if a single row was passed.

A method to check errors is also provided through .calc_err(). It takes a pandas dataframe with a column for the label and a second column title 'pred'. It returns the % of incorrect predictions.

## Bagged Trees
An implementation of bagged trees algorithm using entropy as a purity method.

### How to Use
bagged_trees.py can be ran directly with `python bagged_tree.py` doing so will result in a small mock dataset being generated and used.

To use it externally you need to use the provided constructor for the BaggedTree class. The constructor has 3 required parameters. A pandas dataframe of the data to be trained on, a map mapping attribute names to acceptable values, and a string for the label to be predicted. 

From there there are two options to run the algorithm. You can repeatedly call .run_single() and have it run a single iteration of the algorithm generating a single tree at a time. Or you can just call .run(), this method has an optional parameter T that controlls how many iterations are ran. T defaults to 500. 

Once your trees are generated you can classify data by calling .classify() and either passing a single row or a pandas dataframe. It will return a series of labels if a dataframe was passed and a single value if a single row was passed.

A method to check errors is also provided through .calc_err(). It takes a pandas dataframe with a column for the label and a second column title 'pred'. It returns the % of incorrect predictions.

## Random Forest
An implementation of the Random Forest algorithm using entropy as a purity method.

### How to Use
random_forest.py can be ran directly with `python random_forest.py` doing so will result in a small mock dataset being generated and used.

To use it externally you need to use the provided constructor for the RandomForest class. The constructor has 3 required parameters. A pandas dataframe of the data to be trained on, a map mapping attribute names to acceptable values, and a string for the label to be predicted. 

From there there are two options to run the algorithm. You can repeatedly call .run_single() and have it run a single iteration of the algorithm generating a single tree at a time. Or you can just call .run(), this method has an optional parameter T that controlls how many iterations are ran. T defaults to 500. 

Once your trees are generated you can classify data by calling .classify() and either passing a single row or a pandas dataframe. It will return a series of labels if a dataframe was passed and a single value if a single row was passed.

A method to check errors is also provided through .calc_err(). It takes a pandas dataframe with a column for the label and a second column title 'pred'. It returns the % of incorrect predictions.
