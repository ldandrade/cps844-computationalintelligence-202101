import numpy as np
import matplotlib.pyplot as plt
import common

def assemble_data_set_w_noise(lower_bound, upper_bound, training_points):
    #"(...) Consider the target function \(f(x_1,x_2)=sign(x^2_1+x^2_2-0.6)\) (...) Generate a training set of N = 1000 points on \(X=[-1,1] \times [-1,1]\) with a uniform probability (...)"
    X = np.transpose(np.array([np.ones(training_points), common.random_points(lower_bound, upper_bound, training_points), common.random_points(lower_bound, upper_bound, training_points)]))
    y = np.sign(np.multiply(X[:,1], X[:,1]) + np.multiply(X[:,2], X[:,2]) - 0.6)

    #"(...) Generate simulated noise by flipping the sign of the output in a randomly selected \(10\%\) subset of the generated training set. (...)"
    indices = list(range(training_points))
    np.random.shuffle(indices)
    random_indices = indices[:(training_points // 10)]
    for i in random_indices:
        y[i] = (-1) * y[i]
    return (X,y)

#Question 10
def linear_regression_w_noise_experiment(runs, X, y):
    E_in_total = 0

    for run in range(runs):
        #"(...) Carry out Linear Regression without transformation, i.e., with feature vector \( (1,x_1,x_2) \) to find the weight \(w\) (...)"
        X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)

        #"(...) What is the closest value to the classification in-sample error \(E_{in}\)?"
        y_lr = np.sign(np.dot(X, w))
        E_in = sum(y_lr != y)  / training_points
        E_in_total += E_in
    #"(...) Run the experiment 1000 times and take the average to reduce variation in your results (...)"
    E_in_avg = E_in_total / runs
    return E_in_avg

#Question 11
def linear_regression_w_nonlinear_transformation_experiment(runs, training_points, X, y):
    X_trans = np.transpose(np.array([np.ones(training_points), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))

    #"(...) Find the vector ~w that corresponds to the solution of Linear Regression. (...)"
    X_dagger_trans = np.dot(np.linalg.inv(np.dot(X_trans.T, X_trans)), X_trans.T)
    w_trans = np.dot(X_dagger_trans, y)

    #"(...) Which of the following hypotheses is closest to the one you find? (...)"
    w_a = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
    w_b = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
    w_c = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
    w_d = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
    w_e = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])

    y_lr_trans = np.sign(np.dot(X_trans, w_trans))
    y_a = np.sign(np.dot(X_trans, w_a))
    y_b = np.sign(np.dot(X_trans, w_b))
    y_c = np.sign(np.dot(X_trans, w_c))
    y_d = np.sign(np.dot(X_trans, w_d))
    y_e = np.sign(np.dot(X_trans, w_e))

    #"(...) Closest here means agrees the most with your hypothesis (...)"
    mismatch_lr_and_a = sum(y_a != y_lr_trans) / training_points                                                      
    mismatch_lr_and_b = sum(y_b != y_lr_trans) / training_points
    mismatch_lr_and_c = sum(y_c != y_lr_trans) / training_points
    mismatch_lr_and_d = sum(y_d != y_lr_trans) / training_points
    mismatch_lr_and_e = sum(y_e != y_lr_trans) / training_points

    return (w_trans, mismatch_lr_and_a, mismatch_lr_and_b, mismatch_lr_and_c, mismatch_lr_and_d, mismatch_lr_and_e)

#Question 12
def linear_regression_w_nonlinear_transformation_experiment_evaluation(runs, X_test, y_test, w):
    E_out_total = 0

    for run in range(runs):
        # Compute classification made by hypothesis with weight vector w
        X_trans_test = np.transpose(np.array([np.ones(testing_points), X_test[:,1], X_test[:,2], X_test[:,1]*X_test[:,2], X_test[:,1]*X_test[:,1], X_test[:,2]*X_test[:,2]]))
        y_lr_trans_test = np.sign(np.dot(X_trans_test, w))
    
        # Compute disagreement between hypothesis and target function
        E_out = sum(y_lr_trans_test != y_test) / testing_points
        E_out_total += E_out
    
    E_out_avg = E_out_total / runs
    return E_out_avg

runs = 1000
training_points = 1000
testing_points = 1000

X, y = assemble_data_set_w_noise(-1, 1, training_points)

E_in_avg = linear_regression_w_noise_experiment(runs, X, y)
print("The average error \(E_{in}\) over \(", runs, "\) runs is: \(", E_in_avg,"\)")

w_trans, mismatch_lr_and_a, mismatch_lr_and_b, mismatch_lr_and_c, mismatch_lr_and_d, mismatch_lr_and_e = linear_regression_w_nonlinear_transformation_experiment(runs, training_points, X, y)
print("The weight vector of the hypothesis is: ", w_trans)
print("mismatch between LR and a = ", mismatch_lr_and_a)
print("mismatch between LR and b = ", mismatch_lr_and_b)
print("mismatch between LR and c = ", mismatch_lr_and_c)
print("mismatch between LR and d = ", mismatch_lr_and_d)
print("mismatch between LR and e = ", mismatch_lr_and_e)

# Use that weight vector for problem 10
# compare predictions made by w_lr_trans with those made by targer function
X_test, y_test = assemble_data_set_w_noise(-1, 1, testing_points)
E_out_avg = linear_regression_w_nonlinear_transformation_experiment_evaluation(runs, X_test, y_test, w_trans)
print("The average error \(E_{out}\) over \(", runs, "\) runs is: \(", E_out_avg,"\)")