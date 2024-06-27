import numpy as np

class dense_layer:
    '''
    Dense layer with ReLu activation function
    '''
    def __init__(self,input_dim,output_dim) -> None:
        self.weights = np.random.randn(output_dim,input_dim)
        self.biases = np.random.randn(output_dim,1)

    def forward(self,input):
        self.input = input
        self.z = np.matmul(self.weights,input) + self.biases
        return self.relu(self.z)

    def backward(self,dA):
        m = self.input.shape[0]
        self.dz = dA * self.relu_derivative(self.z)
        self.dw = np.matmul(self.dz,self.input.T) / m
        self.db = np.sum(self.dz) / m
        self.dA_prev = np.matmul(self.weights.T,self.dz)
        return self.dA_prev

    def update_params(self,learning_rate):
        self.weights -= self.dw * learning_rate
        self.biases -= self.db * learning_rate

    def relu(self,x):
        return np.maximum(0,x)

    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

class my_nn:
    def __init__(self,input_size,ouptput_size,learing_rate = 0.01):
        self.input_layer = dense_layer(input_size,5)
        self.output_layer = dense_layer(5,ouptput_size)
        self.learning_rate = learing_rate

    def predict(self,input):
        middle_layer = self.input_layer.forward(input)
        output = self.output_layer.forward(middle_layer)
        return output

    def train(self,X,y,epochs = 10000):
        for _ in range(epochs):
            y_hat = self.predict(X)
            loss = np.mean(np.square(y-y_hat))
            loss_derivative = -(y-y_hat) / len(X)
            middle = self.output_layer.backward(loss_derivative)
            first = self.input_layer.backward(middle)
            self.output_layer.update_params(self.learning_rate)
            self.input_layer.update_params(self.learning_rate)

            if _ % 1000 == 0:
                print(f'Epoch {_}, Loss -> ',loss)
