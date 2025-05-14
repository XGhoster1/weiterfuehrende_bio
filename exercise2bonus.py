import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_points(filename: str ) -> np.array:
    # load 10000 points with ___ coordinates
    points = np.load(f"{filename}")
    return points

def load_labels(filename: str) -> np.array:
    # the labels corresponding to the points: 0 or 1
    labels = np.load(f"{filename}")
    return labels

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
        self.w0 = np.random.randn(100,2)
        self.w1 = np.random.randn(10,100)
        self.w2 = np.random.randn(2,10)
        
        self.learning_rate = 0.05

    

    def activation(self, x):
        return expit(x)

    def trainWeights(self, X, y, learning_rate=0.05):
        # We add bias to input X
        X_b = self.add_bias(X.T)                        

        # We forward pass with bias
        z0 = self.w0 @ X_b                              
        a0 = self.activation(z0)
        a0_b = self.add_bias(a0)                        

        z1 = self.w1 @ a0_b                             
        a1 = self.activation(z1)
        a1_b = self.add_bias(a1)                        

        z2 = self.w2 @ a1_b                             
        pred = self.activation(z2)

        # Backprop
        delta2 = (y.T - pred) * pred * (1 - pred)       
        dw2 = delta2 @ a1_b.T / X.shape[0]              

        error1 = self.w2.T @ delta2                     
        delta1 = error1[1:, :] * a1 * (1 - a1)          
        dw1 = delta1 @ a0_b.T / X.shape[0]              

        error0 = self.w1.T @ delta1                    
        delta0 = error0[1:, :] * a0 * (1 - a0)         
        dw0 = delta0 @ X_b.T / X.shape[0]               

        # Update weights
        self.w2 += learning_rate * dw2
        self.w1 += learning_rate * dw1
        self.w0 += learning_rate * dw0



    def predict(self, X):
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

X_train, X_test, y_train, y_test = train_test_split(
    points, labels, stratify=labels, random_state=42
)

oh = OneHotEncoder()
y_train_oh = oh.fit_transform(y_train.reshape(-1, 1)).toarray()

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

plt.plot(epoche, train_acc, label="Train Accuracy")
plt.plot(epoche, test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("exercise2.pdf")