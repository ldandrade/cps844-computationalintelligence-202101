import numpy as np
import matplotlib.pyplot as plt
import common

def linear_regression_experiment(runs, training_points, test_points):
    E_in_total = 0
    E_out_total = 0

    # repeat the experiment 1000 times
    for run in range(runs):
        #"In each run choose a random line in the plane as your target function f (...)"
        f = common.target_function(lower_bound, upper_bound, dimension)
        X, y = common.assemble_data_set(lower_bound, upper_bound, dimension, training_points, f)
    
        # LINEAR REGRESSION
        X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        lr = np.dot(X_dagger, y)
    
        #Classification of points according to linear regression
        y_lr = np.sign(np.dot(X, lr))
    
        # In-sample error: E_in
        E_in = sum(y_lr != y) / training_points
        E_in_total += E_in

        #"(...) Now generate 1000 fresh points and use them to estimate the out-of-sample error E_out of g (...)"
        X_test, y_test = common.assemble_data_set(lower_bound, upper_bound, dimension, test_points, f)
        
        #Classification of points according to linear regression
        y_lr_test = np.sign(np.dot(X_test, lr))

        # Oout-of-sample error: E_out
        E_out = sum(y_lr_test != y_test) / test_points
        E_out_total += E_out    
    
    # Average of E_in over RUNS
    E_in_avg = E_in_total / runs

    # Average of E_out over RUNS
    E_out_avg = E_out_total / runs

    return (E_in_avg, E_out_avg) 

def pla_initialized_by_linear_regression_experiment(runs, training_points):
    iterations_total = 0

    # repeat the experiment 1000 times
    for run in range(runs):
        #"In each run choose a random line in the plane as your target function f (...)"
        f = common.target_function(lower_bound, upper_bound, dimension)
        X, y = common.assemble_data_set(lower_bound, upper_bound, dimension, training_points, f)
    
        # LINEAR REGRESSION
        X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)
    
        #Classification of points according to linear regression
        #y_lr = np.sign(np.dot(X, lr))

        iterations_count = 0

        #Start the Perceptron Learning Algorithm (PLA)
        while True:
            y_w = np.sign(np.dot(X, w))
            comp = (y_w != y)
            wrong = np.where(comp)[0]

            if wrong.size == 0:
                break
                
            rnd_choice = np.random.choice(wrong)
            w = w +  y[rnd_choice] * np.transpose(X[rnd_choice])
            iterations_count += 1

        iterations_total += iterations_count

    iterations_avg = iterations_total / runs
    return iterations_avg

#Target function f(x) parameters
dimension = 2
lower_bound = -1
upper_bound = 1

#Experiment parameters
runs = 1000
training_points = 100 #sample size
testing_points = 1000

experiment1 = linear_regression_experiment(runs, training_points, testing_points)
print("Average of E_in over", runs, " runs:", experiment1[0])
print("Average of E_out over", runs, " runs:", experiment1[1])

training_points = 10 #sample size
iterations_avg = pla_initialized_by_linear_regression_experiment(runs, training_points)
print("\nAverage number of PLA iterations over \(", runs, "\) runs: \(", iterations_avg, "\)")