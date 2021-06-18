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

#def plot_points(X, y):
    # plot points and color them according to their classification
    #plt.plot(X[:,1][y == 1], X[:,2][y == 1], 'ro')
    #plt.plot(X[:,1][y == -1], X[:,2][y == -1], 'bo')

    # plot line
    # create some data points on the line (for the plot) using the parametric vector form of a line
    # line(t) = A + t * d,  where A is a point on the line, d the directional vector and t the parameter
    #d = B - A
    #line_x = [A[0] + t * d[0] for t in range(-10,10)]
    #line_y = [A[1] + t * d[1] for t in range(-10,10)]
    #plt.plot(line_x, line_y)

    # plot the two points that define the line
    #plt.plot(A[0], A[1], 'go')            
    #plt.plot(B[0], B[1], 'go')


    # set the ranges for the x and y axis to display the [-1,1] x [-1,1] box
    #plt.ylim(-1,1)
    #plt.xlim(-1,1)
    #plt.show()

#def plot_linear_regression():
    # LINEAR REGRESSION
    #X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    #w_lr = np.dot(X_dagger, y_f)

    # plot classification according to w found by linear regression
    # it shows that some of the points are missclassified
    #y_lr = np.sign(np.dot(X, w_lr))
    #print("check dimensions of y_lr: ", y_lr.shape)

    # plot points and color them according to their classification
    #plt.plot(X[:,1][y_lr == 1], X[:,2][y_lr == 1], 'ro')
    #plt.plot(X[:,1][y_lr == -1], X[:,2][y_lr == -1], 'bo')

    # plot the correct classification line (target function)
    #plt.plot(line_x, line_y, 'g')
    #plt.ylim(-1,1)
    #plt.xlim(-1,1)

    #plt.show()