import numpy as np
import matplotlib.pyplot as plt
import common

#Questions 1, 2, 3 and 4
def experiment(runs, training_points):
    #Support variables/counters initialization
    iterations_total = 0
    ratio_mismatch_total = 0

    #Loop of <runs> experiments
    for run in range(runs):
        #"In each run choose a random line in the plane as your target function f (...)"
        f = common.target_function(lower_bound, upper_bound, dimension)
        #"(...) Choose the inputs x of the data set as random points uniformly in X and evaluate the target function on each x to get the corresponding output y (...)"
        X, y = common.assemble_data_set(lower_bound, upper_bound, dimension, training_points, f)
        #"(...) Start the PLA with the weight vector being all zeros (...) so all points are initially misclassified points. (...)"
        h = np.zeros(3)

        iterations_count = 0

        #Start the Perceptron Learning Algorithm (PLA)
        while True:
            y_h = np.sign(np.dot(X, h))
            comp = (y_h != y)
            wrong = np.where(comp)[0]

            if wrong.size == 0:
                break
                
            rnd_choice = np.random.choice(wrong)
            h = h +  y[rnd_choice] * np.transpose(X[rnd_choice])
            iterations_count += 1

        iterations_total += iterations_count

        #Points to calculate E_out
        N_outside = 1000
        test_x0 = np.random.uniform(-1,1,N_outside)
        test_x1 = np.random.uniform(-1,1,N_outside)

        X = np.array([np.ones(N_outside), test_x0, test_x1]).T

        y_target = np.sign(X.dot(f))
        y_hypothesis = np.sign(X.dot(h))

        common.plot_points(X,y_target,f,h)

        ratio_mismatch = ((y_target != y_hypothesis).sum()) / N_outside
        ratio_mismatch_total += ratio_mismatch

    iterations_avg = iterations_total / runs
    ratio_mismatch_avg = ratio_mismatch_total / runs
    return (iterations_avg, ratio_mismatch_avg)

#Question 5
def plot_experiment():
    print("TODO")

#Parameters initialization ------------------------------------

#Target function f(x) parameters
dimension = 2
lower_bound = -1
upper_bound = 1

#Experiment parameters
runs = 1000
training_points = 10

#Experiments execution ----------------------------------------

questions1and2 = experiment(runs, training_points)
print("Size of training data: \(N = ", training_points, "\) points")
print("\nAverage number of PLA iterations over \(", runs, "\) runs: \(", questions1and2[0], "\)")
print("\nAverage ratio for the mismatch between \(f(x)\) and \(h(x)\) outside of the training data:")
print("\(P(f(x) \\neq h(x)) = ", questions1and2[1],"\)")

#Increase size of data set from 10 to 100 points
training_points = 100

#and rerun the previous experiment
questions3and4 = experiment(runs, training_points)
print("Size of training data: \(N = ", training_points, "\) points")
print("\nAverage number of PLA iterations over \(", runs, "\) runs: \(", questions3and4[0],"\)")
print("\nAverage ratio for the mismatch between \(f(x)\) and \(h(x)\) outside of the training data:")
print("\(P(f(x) \\neq h(x)) = ", questions3and4[1], "\)")