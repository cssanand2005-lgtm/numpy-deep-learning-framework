# 🧠 Custom Deep Learning Framework in Pure NumPy

A PyTorch-like deep learning library built completely from scratch using **only NumPy** (zero high-level deep learning frameworks). This library implements forward passes, backpropagation via exact analytical derivatives, activation functions, loss functions, and optimization algorithms from first principles.

---

## 🌟 Key Features

- **Modular Architecture**: Base `Layer` abstraction class mirroring PyTorch's `nn.Module`.
- **Custom Layers & Activations**: `Linear` (Dense fully connected layer) and `ReLU` non-linear activation.
- **Loss Functions**: `SoftmaxCrossEntropy` combined for numerical stability and simplified gradients.
- **Optimizers**: Custom `SGD` (Stochastic Gradient Descent) optimizer with configurable learning rate.
- **Sequential Container**: `Sequential` wrapper to stack layers and handle automated forward/backward propagation loops.
- **Interactive Web App**: Streamlit web dashboard for adjusting hyperparameters in real-time and visualizing decision boundaries.

---

## 📁 Repository Structure

```text
numpy-deep-learning-framework/
│
├── mini_torch/              # Core deep learning engine library
│   ├── __init__.py          # Package initializer
│   ├── layers.py            # Linear & ReLU layer definitions
│   ├── losses.py            # Softmax & Cross Entropy Loss
│   ├── optimizers.py        # SGD optimizer implementation
│   └── container.py         # Sequential model container
│
├── train_spiral.py          # Command-line training script
├── app.py                   # Streamlit interactive web interface
└── README.md                # Documentation
