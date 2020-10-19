import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return (1 - sigmoid(x)) * sigmoid(x)

class NeuralNetwork:

    def __init__(self, layers, alpha = 0.1):

        self.layers = layers

        #learning rate
        self.alpha = alpha

        #Khoi tao W
        self.W = []

        #Khoi tao b
        self.b = []

        #Khoi tao bo trong so:
        for i in range(0, len(layers)-1):
            W_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1], 1))
            self.W.append(W_/layers[i])
            self.b.append(b_)

    #Model Summarize
    def __repr__(self):
        return "Neural Network [{}]".format("-".join(str(l) for l in self.layers))

    #Train model
    def train(self, X, y):
        A = [X]
        self.layers = layers

        #feed forward
        out = A[-1]
        for i in range(0, len(layers)-1):
            out = np.dot(out, self.W[i]) + self.b[i].T
            out = sigmoid(out)
            A.append(out)

        #back-propagation
        y = y.reshape(-1,1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []

        for i in reversed(0, len(layers) - 1):
            dw_ = np.dot((A[i].T, dA[-1]*sigmoid_derivative(A[i+1])))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        #Reverse W,b
        dW = dW[::-1]
        db = db[::-1]

        #Gradient descent
        for i in range (0, len(layers) -1 ):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def fit(self, X, y, epochs = 20, verbose = 10):

        for i in range (0, epochs):
            self.fit_partial(X,y)

            if epochs % verbose ==0:
                loss = self.caculate_loss(X,y)
                print("Epoch: {}, Loss: {}".format(Epochs, loss))

    def predict(self,X):
        for i in range (0, len(self.layers) - 1):
            X = sigmoid(np.sum(np.dot(X, self.W[i])), self.b[i].T)
            return X

    def caculate_loss(sefl, X, y):
        y_pred = self.predict(X)
        #loss 1: binary crossentropy
        return -(np.sum(y*np.log(y_pred)) + (1-y) * (1 - np.log(1 - y_pred)))
        
        #loss 2
        #for i in range (0, len(self.layers)-1):
            #element_loss = 











       


