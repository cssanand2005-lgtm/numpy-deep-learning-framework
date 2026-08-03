import numpy as np
class Layer:
    def __init__(self):
        self.params, self.grads = {}, {}
class Linear(Layer):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.params["W"] = np.random.randn(in_f, out_f) * np.sqrt(2.0 / in_f)
        self.params["b"] = np.zeros((1, out_f))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
    def forward(self, x):
        self.x = x
        return np.dot(x, self.params["W"]) + self.params["b"]
    def backward(self, g):
        self.grads["W"] = np.dot(self.x.T, g)
        self.grads["b"] = np.sum(g, axis=0, keepdims=True)
        return np.dot(g, self.params["W"].T)
class ReLU(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    def backward(self, g):
        return g * (self.x > 0)
