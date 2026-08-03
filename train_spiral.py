import numpy as np
from mini_torch import Linear, ReLU, SoftmaxCrossEntropy, SGD, Sequential

# 1. Generate Synthetic Data
np.random.seed(42)
X = np.random.randn(300, 2)
y = np.random.randint(0, 3, 300)

# 2. Build Custom Neural Network Architecture
model = Sequential([
    Linear(2, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 3)
])

criterion = SoftmaxCrossEntropy()
optimizer = SGD(model.layers, lr=0.1)

# 3. Training Loop
print("--- Training Custom Neural Network ---")
for epoch in range(501):
    logits = model.forward(X)
    loss = criterion.forward(logits, y)
    
    loss_grad = criterion.backward()
    model.backward(loss_grad)
    optimizer.step()
    
    if epoch % 100 == 0:
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

print("\nSUCCESS: Custom Neural Network trained successfully!")