import csv
import numpy as np
from scipy.optimize import minimize
from math import sqrt

data = csv.reader(open('sample_ratings_copy.csv', 'rb'), delimiter=",")
books, users, ratings = [], [], []
for row in data:
    books.append(row[0])
    users.append(row[1])
    ratings.append(row[2])

books = map(int, books)
users = map(int, users)
ratings = map(int, ratings)

Y = np.zeros(shape=(20, 7))
R = np.zeros(shape=(20, 7))
for i in range(72):
  Y[books[i] - 1, users[i] - 1] = ratings[i]
  R[books[i] - 1, users[i] - 1] = 1


#training the collaborative filtering model
num_users = Y.shape[1]
num_books = Y.shape[0]
num_features = 50 #to be selected using learning curve
learning_rate = 0.5
Y = np.matrix(Y)
R = np.matrix(R)

#initialize feature vector and parameter vector
X = np.random.random(size=(num_books, num_features))
Theta = np.random.random(size=(num_users, num_features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

#normalize Y
Ymean = np.zeros((num_books, 1))
Ynorm = np.zeros((num_books, num_users))

for i in range(num_books):
    idx = np.where(R[i,:] == 1)[0]
    # Ymean[i] = Y[i,idx].mean()
    # Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    Ynorm[i,idx] = Y[i,idx] / 5

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
    J = (1 / 2) * np.sum(sq_error) + reg_term
    print J
    #gradients
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)
    #unravel X_grad and Theta_grad into grad
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

# J, grad = cost(params, Ynorm, R, num_features, learning_rate)

#optimization
minimized = minimize(fun=cost, x0=params, args=(Ynorm, R, num_features, learning_rate), method='BFGS', jac=True, options={'maxiter': 100})
X = (np.reshape(minimized.x[:num_books * num_features], (num_books, num_features)))
Theta = (np.reshape(minimized.x[num_books * num_features:], (num_users, num_features)))

#prediction matrix
predictions = np.zeros(shape=(num_books, num_users))
X = np.matrix(np.reshape(params[:num_books * num_features], (num_books, num_features)))
Theta = np.matrix(np.reshape(params[num_books * num_features:], (num_users, num_features)))
predictions = X * Theta.T

#prediction accuracy
p = np.multiply(predictions, R)
p = np.array(p)
p = p.astype(int)
for i  in range(20):
    for j in range(7):
        if(p[i, j] > 5):
            p[i, j] = 5

accuracy_percent = np.mean(p == Y) * 100
print p
print Y
print accuracy_percent

#recommend books
