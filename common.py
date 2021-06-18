import numpy as np
import matplotlib.pyplot as plt

def random_points(lower_bound, upper_bound, dimension):
    return np.random.uniform(lower_bound, upper_bound, size = dimension)

def target_function(lower_bound, upper_bound, dimension):
    # choose two random points A and B that belong to domain X
    A = random_points(lower_bound, upper_bound, dimension)
    B = random_points(lower_bound, upper_bound, dimension)
    
    # a line passing through 2 points A and B can be described by y = m*x + b where m is the slope
    # where m = y - b / x
    # and b = y - m*x
    m = (B[1] - A[1]) / (B[0] - A[0])
    b = B[1] - m * B[0]  
    return np.array([b, m, -1])

def assemble_data_set(lower_bound, upper_bound, dimension, sample_size, target_function):
    #"(...) Choose the inputs x of the data set as random points uniformly in X and evaluate the target function on each x to get the corresponding output y (...)"
    X = np.transpose(np.array([np.ones(sample_size), random_points(lower_bound, upper_bound, sample_size), random_points(lower_bound, upper_bound, sample_size)]))           # input
    #np.sign is used to map one side of the line to -1 and the other to +1, hence giving the corresponding output y to an input x
    y = np.sign(np.dot(X, target_function))
    return (X, y)