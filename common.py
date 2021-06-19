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

def assemble_data_set_w_noise(lower_bound, upper_bound, training_points, generate_target_function):
    #"(...) Consider the target function \(f(x_1,x_2)=sign(x^2_1+x^2_2-0.6)\) (...) Generate a training set of N = 1000 points on \(X=[-1,1] \times [-1,1]\) with a uniform probability (...)"
    X = np.transpose(np.array([np.ones(training_points), random_points(lower_bound, upper_bound, training_points), random_points(lower_bound, upper_bound, training_points)]))
    if generate_target_function:
        f = target_function(-1,1,2)
        y = np.sign(np.dot(X, f))
    else:
        f = []
        y = np.sign(np.multiply(X[:,1], X[:,1]) + np.multiply(X[:,2], X[:,2]) - 0.6)

    #"(...) Generate simulated noise by flipping the sign of the output in a randomly selected \(10\%\) subset of the generated training set. (...)"
    indices = list(range(training_points))
    np.random.shuffle(indices)
    random_indices = indices[:(training_points // 10)]
    for i in random_indices:
        y[i] = (-1) * y[i]
    return (X,y,f)

def plot_points(X, y, f, h):
    # set the ranges for the x and y axis to display the [-1,1] x [-1,1] box
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    axes = plt.gca()

    #plot points and color them according to their classification
    plt.plot(X[:,1][y == 1], X[:,2][y == 1], 'ro')
    plt.plot(X[:,1][y == -1], X[:,2][y == -1], 'bo')

    #y = m*x + b
    #f = [b,m,-1]
    x_vals = np.array(axes.get_xlim())
    y_vals = f[0] + f[1] * x_vals

    #Plot the target function
    plt.plot(x_vals, y_vals, 'black')

    #Plot the hypothesis function
    yh_vals = h[0] + h[1] * x_vals
    plt.plot(x_vals, yh_vals, 'g')

    plt.show()