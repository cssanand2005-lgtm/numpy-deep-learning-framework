import numpy as np
class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        self.y = y
        N = logits.shape[0]
        return -np.sum(np.log(self.probs[np.arange(N), y] + 1e-15)) / N
    def backward(self):
        N = self.probs.shape[0]
        g = self.probs.copy()
        g[np.arange(N), self.y] -= 1
        return g / N
