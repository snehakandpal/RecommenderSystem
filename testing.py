import csv
import numpy as np
from scipy.optimize import minimize
from math import sqrt

data = csv.reader(open('ratings.csv', 'rb'), delimiter=",")
books, users, ratings = [], [], []
for row in data:
    users.append(row[0])
    books.append(row[1])
    ratings.append(row[2])

books = map(int, books)
users = map(int, users)
ratings = map(int, ratings)
print "mapped"

Y = np.zeros(shape=(10000, 53424))
R = np.zeros(shape=(10000, 53424))
Q = np.ones(shape=(10000, 53424))

for i in range(5976479):
  Y[books[i] - 1, users[i] - 1] = ratings[i]
  R[books[i] - 1, users[i] - 1] = 1
  Q[books[i] - 1, users[i] - 1] = 0
print "n-d array"
#training the collaborative filtering model
num_users = Y.shape[1]
num_books = Y.shape[0]
num_features = 10  #to be selected using learning curve
learning_rate = 1.5
Y = np.matrix(Y)
R = np.matrix(R)
Q = np.matrix(Q)
print "matrix created"
#initialize feature vector and parameter vector
X = np.random.random(size=(num_books, num_features))
Theta = np.random.random(size=(num_users, num_features))
#initial_parameters = np.concatenate((np.ravel(X), np.ravel(Theta)))
#X = np.matrix(X)
#Theta = np.matrix(Theta)
params = np.concatenate((np.ravel(X), np.ravel(Theta)))
#cost function and optimization
def cost(params, Y, R, num_features, learning_rate):
    #arguments: X Theta Y R num_users num_books num_features learning_rate
    #returns: J X_grad Theta_grad
    #reshape X and Theta
    X = np.matrix(np.reshape(params[:num_books * num_features], (num_books, num_features)))
    Theta = np.matrix(np.reshape(params[num_books * num_features:], (num_users, num_features)))
    #intializations
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    #cost
    error = np.multiply((X * Theta.T) - Y, R)
    sq_error = np.power(error, 2)
    reg_term = (learning_rate / 2) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))
    J = (np.sum(sq_error) + reg_term) / 2
    print J
    #gradients
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)
    #unravel X_grad and Theta_grad into grad
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

#J, grad = cost(params, Y, R, num_features, learning_rate)
print "minimizing"
#optimization
fmin = minimize(fun=cost, x0=params, args=(Y, R, num_features, learning_rate), method='L-BFGS-B', jac=True, options={'maxiter': 1000})
print "optimized"

#prediction matrix
predictions = np.zeros(shape=(num_books, num_users))
X = np.matrix(np.reshape(params[:num_books * num_features], (num_books, num_features)))
Theta = np.matrix(np.reshape(params[num_books * num_features:], (num_users, num_features)))
predictions = X * Theta.T

# #prediction accuracy
p = np.array(predictions)
p = np.round(p)
# dict = {el:0 for el in range(500)}
#
# for i  in range(10000):
#     for j in range(53424):
#         if p[i, j] < 500:
#             dict[p[i, j]] = dict.get(p[i, j]) + 1
#         else:
#             print ">500"
#
# print dict

np.savetxt("prediction_matrix.csv", p, delimiter=",")

p = np.multiply(p, R)
print "prediction accuracy"
total = 0
equal = 0
for i  in range(10000):
    for j in range(53424):
        if(p[i, j] != 0):
            if(p[i, j] > 5):
                p[i, j] = 5
            total = total + 1
            if(p[i, j] == Y[i, j] or p[i, j] == Y[i, j]-1 or p[i, j] == Y[i, j]+1):
                equal = equal + 1
accuracy = equal * 100 / total
print accuracy

#
# accuracy_percent = np.mean(p == Y) * 100
# print p
# print Y
# print accuracy_percent

# #recommend books
# #to users who have rated
# ptemp = np.multiply(p, Q)
# def find_max(u, pred):
#     n = len(pred)
#     return (-pred).argsort()[:4]
#
# for u in range(7):
#     max_indices = find_max(u, p[:, u])
#     print max_indices
#
# #to users who have only read
