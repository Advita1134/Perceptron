"""
The perceptron is a binary classifier (Supervised learning).

A binary classifier means that it only has two classes.
The perceptron names these classes either -1 or 1.
"""

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

## Data ##
X = [[13], [16], [12], [10], [11], [14], [15], [17], [35], [46], [40], [37], [42], [44], [36], [38]]
y = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]

X_train,X_test,y_train,y_test = train_test_split(X,y) # splitting data

## Model ##
model = Perceptron(max_iter = 10000, eta0 = .5) 

## Training ##
model.fit(X_train,y_train)

## Testing ##
y_pred = model.predict(X_test)

# Checking the results and accuracy after testing.
print(y_pred)
print(y_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# Making a plot using Matplotlib of the training and testing data.
train_zeros = len(X_train) * [0] # Zeros for the y-values of the training data (to convert the 1D data to a 2D graph).
test_zeros = len(X_test) * [0]
plt.scatter(X_train,train_zeros, c = y_train, marker = ".", s = 80, label = "Training Data")
plt.scatter(X_test,test_zeros, c = y_pred, marker = "*", s = 150, label = "Testing Data")

# Making the Decision Line (z = mx + b). This is shown in blue in the figure below.
m = model.coef_[0, 0] # Extracting the slope (that was found in model.fit)
b = model.intercept_[0] # Extracting the y-intercept

maxXvalue = max(X)[0]
xplot = [0, maxXvalue]
yplot = [b ,maxXvalue*m + b]
plt.plot(xplot,yplot, label = "z=mx+b")
plt.ylim([-1, 2])
plt.grid()

# Making the Classification Line to show the split of the data (shown in red below).
xint = -b/m
plt.vlines(xint, -1, 1,  color = "red", linewidth = 5)
plt.hlines(1, xint, max(X), color = "red", linewidth = 5)
plt.hlines(-1, min(X), xint, color = "red", linewidth = 5)
xtext = (max(X) + xint)/2
plt.text(xtext, 1.1, "out", color = "red", fontsize = 25)

plt.legend()
