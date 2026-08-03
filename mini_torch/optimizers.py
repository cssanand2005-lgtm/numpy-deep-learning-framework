class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers; self.lr = lr
    def step(self):
        for l in self.layers:
            for p in l.params: l.params[p] -= self.lr * l.grads[p]
