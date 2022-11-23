# Perceptron
An implementation of the perceptron algorithm allowing for standard, voted, and average methods.

## How to use it
To run the perceptron algorithm simply import it and run it by calling the function for the method as desired. 

The supported methods are .standard, .voted, and .average. Each expects two numpy.ndarray for x and y values. Each also supports optional parameters, lr and T. lr is the learning rate and defaults to 0.1. T is the number of epochs to run and defaults to 10.

driver.py runs an example using the bank-note dataset.