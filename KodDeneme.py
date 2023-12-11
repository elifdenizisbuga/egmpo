import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from keras.datasets.mnist import load_data
from keras.optimizers import Adam

class ParametricReLU():
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.maximum(self.alpha * x, x)

    def backward(self, gradout):
        new_grad = gradout.copy()
        new_grad[self.x < 0] *= self.alpha
        return new_grad

class MLP():
    def __init__(self, din, dout):
        self.W = (2 * np.random.rand(dout, din) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.b = (2 * np.random.rand(dout) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.activation = ParametricReLU()  # Burada ParametricReLU kullanıyoruz.

    def forward(self, x):
        self.x = x
        return self.activation.forward(x @ self.W.T + self.b)

    def backward(self, gradout):
        gradout = self.activation.backward(gradout)
        self.deltaW = gradout.T @ self.x
        self.deltab = gradout.sum(0)
        return gradout @ self.W

class SequentialNN():
    def __init__(self, blocks: list):
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, gradout):
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
        return gradout

class LogSoftmax():
    def forward(self, x):
        self.x = x
        return x - logsumexp(x, axis=1, keepdims=True)

    def backward(self, gradout):
        gradients = np.eye(self.x.shape[1])[None, ...]
        gradients -= np.exp(self.x) / np.sum(np.exp(self.x), axis=1, keepdims=True)
        return np.matmul(gradients, gradout[..., None])[:, :, 0]

class NLLLoss():
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b, true[b]]
        return loss

    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((self.pred.shape[0], din))
        for b in range(self.pred.shape[0]):
            jacobian[b, self.true[b]] = -1
        return jacobian  # batch_size x din

    def __call__(self, pred, true):
        return self.forward(pred, true)

class Optimizer():
    def __init__(self, compound_nn: SequentialNN):
        self.compound_nn = compound_nn
        self.optimizer = Adam()  # Burada Adam optimizer kullanıyoruz.

    def step(self):
        for block in self.compound_nn.blocks:
            if block.__class__ == MLP:
                block.W = self.optimizer.get_updates(block.W, self.optimizer.get_gradients(None, block.W))[0]
                block.b = self.optimizer.get_updates(block.b, self.optimizer.get_gradients(None, block.b))[0]

def train(model, optimizer, trainX, trainy, loss_fct=NLLLoss(), nb_epochs=14000, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        batch_idx = np.random.choice(trainX.shape[0], batch_size, replace=False)
        x = trainX[batch_idx]
        target = trainy[batch_idx]
        prediction = model.forward(x)
        loss_value = loss_fct(prediction, target)
        training_loss.append(loss_value)
        gradout = loss_fct.backward()
        model.backward(gradout)
        optimizer.step()
    return training_loss

if __name__ == "__main__":
    # Your data loading and preprocessing here
    # Load your CSV data and preprocess it accordingly
    # Assume that you have trainX, trainy, testX, testy defined

    # Update the model architecture to match your requirements
    mlp = SequentialNN([
        MLP(input_size, 128), ParametricReLU(),  # Adjust input_size based on your CSV data
        MLP(128, 64), ParametricReLU(),
        MLP(64, 375), LogSoftmax()  # Assuming 375 output classes
    ])

    optimizer = Optimizer(mlp) 

    # Train the model
    training_loss = train(mlp, optimizer, trainX, trainy)

    # Evaluate the model
    accuracy = 0
    for i in range(testX.shape[0]):
        prediction = mlp.forward(testX[i].reshape(1, -1)).argmax()
        if prediction == testy[i]:
            accuracy += 1
    print('Test accuracy', accuracy / testX.shape[0] * 100, '%')
