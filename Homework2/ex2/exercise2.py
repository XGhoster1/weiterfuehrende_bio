import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def load_points(filename: str ) -> np.array:
    # load 10000 points with ___ coordinates
    ____ 

def load_labels(filename: str) -> np.array:
    # the labels corresponding to the points: 0 or 1
    ____

#
#  first: ignore b (offset/bias)
#  second: when everything works think about to add b to your data
#  i.e. pad X values with on, add weight for b
#  IMPORTANT: Add b either as first row or first column to the weight matrices


class NeuralNetwork(object):
    def __init__(self):
        # load training and test data from file
        # create and initialize your weigth matrices
        # consider more parameters like learning rate,
        # layer size, batch size, ...
        self.w0 = np.random.randn(____,____)
        self.w1 = np.random.randn(____,____)
        self.w2 = np.random.randn(____,____)

    

    def activation(self, x):
        return expit(x)

    def trainWeights(self, X, y, learning_rate=0.05):
        # apply first layer
        # apply second
        # apply third layer
        # calculate final error (layer 3)
        # back propapate errors to layer 2
        # back propapate errors to layer 1
        # calculate derivates dw, normalize by division with len(X)
        # help: assert dw.shape == self.w.shape
        # update weights of layers
        a0 = self.activation(self.w0 @ X.T)
        a1 = self.activation(___________)
        pred = self.activation(___________)

        # compare true labels to predicted 
        output_error = y.T - pred
        # partial derivatives for layer 3
        # the pattern is this err * pred * (1-pred) @ input from the 
        # previous layer, normalized by the number of points
        dw2 = (_______________________________) @ a1.T / len(X)
        # backpropagate error to layer 2
        # error of the previous layer is error @ weights
        a1_error = output_error.T @ self.________
        # partial derivatives for layer 2      
        dw1 = (_____________________________) @ _______ / len(X)
        # backpropagate error to layer 2
        a0_error = ________ @ self.w1
        # partial derivatives for layer 1
        dw0 = ____________________________________


        assert dw2.shape == self.w2.shape
        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape
        
        
        # update the weights by applying the learning rate to dw
        self.w2 += learning_rate * dw2
        self.w1 += learning_rate * dw1
        self.w0 += learning_rate * _____

    def predict(self, X):
        # apply first layer
        # apply second layer
        # return predictions
        a0 = self.activation(self.w0 @ X.T)
        a1 = self.activation(self.w1 @ a0)
        pred = self.activation(self.w2 @ a1)
        return pred

    def costs(self, predictions, y):
        # calculate mean costs per point
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - predictions) ** 2
        return np.mean(np.sum(s, axis=0))



# load data

labels = load_points("data/exercise2-labels.npy")
points = load_points("data/exercise2-data.npy")

print("labels: " + str(labels.shape))
print("points: " + str(points.shape))

X_train, X_test, y_train, y_test = train_test_split(_______, ________, _______=labels, ________=42)

oh = OneHotEncoder()
y_train_oh = oh.____________(y_train.reshape(-1, 1)).toarray()

model = NeuralNetwork()

epoche = []
train_acc = []
test_acc = []

for i in range(1000):

    model.trainWeights(X_train, y_train_oh, learning_rate=0.08)
    y_test_predictions = model.predict(X_test)
    y_test_predictions = np.argmax(y_test_predictions, axis=0)
    train_predictions = np.argmax(model.predict(X_train), axis=0)
    print("accuracy on test set: " + str(np.mean(y_test_predictions == y_test)) + " costs on training set: " + str(model.costs(train_predictions, y_train)))
    epoche.append(i)
    test_acc.append(np.mean(y_test_predictions == y_test))
    train_acc.append(np.mean(train_predictions == y_train))                

print("baseline: " + str(np.sum(labels)/len(labels)))

plt.plot(epoche, train_acc, test_acc)
plt.show()