# SVM
An implementation of the SVM algorithms allowing for primal, dual, and gaussian kernel methods.

## How to use it
To run the perceptron algorithm simply import it and run it by calling the function for the method as desired. 

The supported methods are .primal, .dual, and .gaussian_kernel. Each expects two numpy.ndarray for x and y values.

The .primal method expects a third parameter of a method with a single input to determine the learning rate at each iteration. Additionally it allows for passing of the C hyperparameter and T epoch count.

The .dual method expects a third parameter for the C hyperparameter. 

The .gaussian_kernel method expects two additional parameters for C and gamma. To get the predictions for this method an additional method is provided via .predict_gk. This prediction  method takes the parameters alpha, train_x, train_y, test_x, gamma. Alphas is the list of alphas provided by the main method, train_x and train_y are the x and y of the training data and test_x is the data to be predicted. Gamma is the gamma used in the initial run of the gaussian_kernel method.

driver.py runs an example using the bank-note dataset. Each individual method can also be called by passing an argument when calling driver of "primal", "dual", or "gk".