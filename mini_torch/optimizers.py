class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers; self.lr = lr
    def step(self):
        for l in self.layers:
            for p in l.params: l.params[p] -= self.lr * l.grads[p]
class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = []
        self.v = []
        for l in self.layers:
            if hasattr(l, "params") and l.params:
                self.m.append({p: np.zeros_like(l.params[p]) for p in l.params})
                self.v.append({p: np.zeros_like(l.params[p]) for p in l.params})
            else:
                self.m.append({})
                self.v.append({})

    def step(self):
        self.t += 1
        for i, l in enumerate(self.layers):
            if not hasattr(l, "params") or not l.params:
                continue
            for p in l.params:
                grad = l.grads[p]
                self.m[i][p] = self.beta1 * self.m[i][p] + (1 - self.beta1) * grad
                self.v[i][p] = self.beta2 * self.v[i][p] + (1 - self.beta2) * (grad**2)

                m_hat = self.m[i][p] / (1 - self.beta1**self.t)
                v_hat = self.v[i][p] / (1 - self.beta2**self.t)

                l.params[p] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
