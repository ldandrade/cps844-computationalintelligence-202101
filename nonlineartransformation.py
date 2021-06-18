#Nonlinear Transformation
import numpy as np
import matplotlib.pyplot as plt

# create 1000 random points
N_train = 1000

def rnd(n):
    return np.random.uniform(-1, 1, size = n)

# matrix consisting of feature vectors
X_train = np.transpose(np.array([np.ones(N_train), rnd(N_train), rnd(N_train)]))
y_f_train = np.sign(np.multiply(X_train[:,1], X_train[:,1]) + np.multiply(X_train[:,2], X_train[:,2]) - 0.6)
print(X_train.shape)
print(y_f_train.shape)


# pick 10% = 100 random indices
indices = list(range(N_train))
np.random.shuffle(indices)
random_indices = indices[:(N_train // 10)]


# flip sign in y_f_train vector
for idx in random_indices:
    y_f_train[idx] = (-1) * y_f_train[idx]

# linear regression
X_dagger = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T)
w_lr_train = np.dot(X_dagger, y_f_train)

# calculate E_in
y_lr_train = np.sign(np.dot(X_train, w_lr_train))
E_in = sum(y_lr_train != y_f_train)  / N_train
print("In sample error: ", E_in)


# Create a plot of the classified points
plt.plot(X_train[:,1][y_f_train == 1], X_train[:,2][y_f_train == 1], 'ro')
plt.plot(X_train[:,1][y_f_train == -1], X_train[:,2][y_f_train == -1], 'bo')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

#Problem 8
# Now do this 1000 times to take average
import numpy as np
import matplotlib.pyplot as plt

def rnd(n):
    return np.random.uniform(-1, 1, size = n)


RUNS = 1000
N_train = 1000
E_in_total = 0

for run in range(RUNS):
    
    # create 1000 random points
    # matrix consisting of feature vectors
    X_train = np.transpose(np.array([np.ones(N_train), rnd(N_train), rnd(N_train)]))
    y_f_train = np.sign(X_train[:,1] * X_train[:,1] + X_train[:,2] * X_train[:,2] - 0.6)

    # pick 10% = 100 random indices
    indices = list(range(N_train))
    np.random.shuffle(indices)
    random_indices = indices[:(N_train // 10)]

    # flip sign in y_f_train vector
    for idx in random_indices:
        y_f_train[idx] = (-1) * y_f_train[idx]

    # linear regression
    X_dagger = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T)
    w_lr_train = np.dot(X_dagger, y_f_train)

    # calculate E_in
    y_lr_train = np.sign(np.dot(X_train, w_lr_train))
    E_in = sum((y_lr_train != y_f_train))  / N_train
    E_in_total += E_in
    #print("In sample error: ", E_in)

    
E_in_avg = E_in_total / RUNS
print("The average error E_in over", RUNS, "runs is: E_in_avg = ", E_in_avg)

# Problem 9 :  transform the N = 1000 training data into the following nonlinear feature
# vector: (1, x1, x2, x1*x2, x1*x1, x2*x2)

# How to concatenate extra columns to X ?
X = X_train

# new feature matrix
X_trans = np.transpose(np.array([np.ones(N_train), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))


# linear regression on the new "feature matrix"
X_dagger_trans = np.dot(np.linalg.inv(np.dot(X_trans.T, X_trans)), X_trans.T)
w_lr_trans = np.dot(X_dagger_trans, y_f_train)

# try the different hypotheses that are given
w_a = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
w_b = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
w_c = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
w_d = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
w_e = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])

# compute classifications made by each hypothesis
y_lr_trans = np.sign(np.dot(X_trans, w_lr_trans))
y_a = np.sign(np.dot(X_trans, w_a))
y_b = np.sign(np.dot(X_trans, w_b))
y_c = np.sign(np.dot(X_trans, w_c))
y_d = np.sign(np.dot(X_trans, w_d))
y_e = np.sign(np.dot(X_trans, w_e))

mismatch_lr_and_a = sum(y_a != y_lr_trans) / N_train                 # ALWAYS RESTART KERNEL !!!!!!!!!!!!                                                         
mismatch_lr_and_b = sum(y_b != y_lr_trans) / N_train
mismatch_lr_and_c = sum(y_c != y_lr_trans) / N_train
mismatch_lr_and_d = sum(y_d != y_lr_trans) / N_train
mismatch_lr_and_e = sum(y_e != y_lr_trans) / N_train

print("mismatch between LR and a = ", mismatch_lr_and_a)
print("mismatch between LR and b = ", mismatch_lr_and_b)
print("mismatch between LR and c = ", mismatch_lr_and_c)
print("mismatch between LR and d = ", mismatch_lr_and_d)
print("mismatch between LR and e = ", mismatch_lr_and_e)

print("The weight vector of my hypothesis is: w_LR = ", w_lr_trans)
# Use that weight vector for problem 10


# compare predictions made by w_lr_trans with those made by targer function
print("Sanity check: E_in = ", sum(y_f_train != y_lr_trans) / N_train)

# Problem 10

RUNS = 1000
N_test = 1000
E_out_total = 0

for run in range(RUNS):
    
    # create 1000 random points
    # matrix consisting of feature vectors
    X_test = np.transpose(np.array([np.ones(N_train), rnd(N_train), rnd(N_train)]))
    y_f_test = np.sign(X_test[:,1] * X_test[:,1] + X_test[:,2] * X_test[:,2] - 0.6)

    # pick 10% = 100 random indices
    indices = list(range(N_test))
    np.random.shuffle(indices)
    random_indices = indices[:(N_test // 10)]

    # flip sign in y_f_train vector
    for idx in random_indices:
        y_f_test[idx] = (-1) * y_f_test[idx]

    # Compute classification made by my hypothesis from Problem 9
    # first create transformed feature matrix
    X = X_test
    X_trans_test = np.transpose(np.array([np.ones(N_test), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))
    y_lr_trans_test = np.sign(np.dot(X_trans_test, w_lr_trans))
    
    # Compute disagreement between hypothesis and target function
    E_out = sum(y_lr_trans_test != y_f_test) / N_train
    E_out_total += E_out
    
E_out_avg = E_out_total / RUNS
print("The average error E_out over", RUNS, "runs is: E_out_avg = ", E_out_avg)