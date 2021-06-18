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

def experiment(runs, ):
    #Support variables/counters initialization
    iterations_total = 0
    ratio_mismatch_total = 0

    #Loop of <runs> experiments
    for run in range(runs):
        #"In each run choose a random line in the plane as your target function f (...)"
        f = target_function(lower_bound, upper_bound, dimension)
        print("Target function f(x) = ", f[1], "x +", f[0])

        #"(...) Choose the inputs x of the data set as random points uniformly in X and evaluate the target function on each x to get the corresponding           #output y (...)"
        X = np.transpose(np.array([np.ones(training_points), random_points(lower_bound, upper_bound, training_points), random_points(lower_bound, upper_bound, training_points)]))
        print(X)

        #np.sign is used to map one side of the line to -1 and the other to +1, hence giving the corresponding output y to an input x
        #np.dot is used to 
        y = np.sign(np.dot(X, f))

        #"(...) Start the PLA with the weight vector being all zeros (...) so all points are initially misclassified points. (...)"
        h = np.zeros(3)

        iterations_count = 0

        #Start the Perceptron Learning Algorithm (PLA)
        while True:
            y_h = np.sign(np.dot(X, h))       # classification by hypothesis
            comp = (y_h != y)                 # compare classification with actual data from target function
            wrong = np.where(comp)[0]           # indices of points with wrong classification by hypothesis h

            if wrong.size == 0:
                break
                
            rnd_choice = np.random.choice(wrong)        # pick a random misclassified point

            # update weight vector (new hypothesis):
            h = h +  y[rnd_choice] * np.transpose(X[rnd_choice])
            iterations_count += 1

        iterations_total += iterations_count

        #Out-of-sample Data

        # Calculate error
        # Create data "outside" of training data

        N_outside = 1000
        test_x0 = np.random.uniform(-1,1,N_outside)
        test_x1 = np.random.uniform(-1,1,N_outside)

        X = np.array([np.ones(N_outside), test_x0, test_x1]).T

        y_target = np.sign(X.dot(f))
        y_hypothesis = np.sign(X.dot(h))

        ratio_mismatch = ((y_target != y_hypothesis).sum()) / N_outside
        ratio_mismatch_total += ratio_mismatch

        return iterations_total, ratio_mismatch_total

#Parameters initialization

#Target function f(x) parameters
dimension = 2
lower_bound = -1
upper_bound = 1

#Experiment parameters
runs = 1000

training_points = 10
experiment1 = experiment(runs, training_points)

training_points = 100
experiment2 = experiment(runs, training_points)