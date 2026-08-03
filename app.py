import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from mini_torch import Linear, ReLU, SoftmaxCrossEntropy, SGD, Sequential

st.set_page_config(page_title="NumPyTorch Interactive Engine", layout="wide")

st.title("🧠 Custom Neural Network Engine (NumPy)")
st.write("A deep learning framework built completely from scratch using **only NumPy**.")

st.sidebar.header("Hyperparameters")
hidden_units = st.sidebar.slider("Hidden Units", min_value=16, max_value=128, value=64, step=16)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=1000, value=500, step=100)

def create_spiral_data(samples_per_class=100, classes=3):
    X = np.zeros((samples_per_class * classes, 2))
    y = np.zeros(samples_per_class * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples_per_class * class_number, samples_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, samples_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
    return X, y

X, y = create_spiral_data()

if st.button("🚀 Train Model"):
    model = Sequential([
        Linear(2, hidden_units),
        ReLU(),
        Linear(hidden_units, hidden_units),
        ReLU(),
        Linear(hidden_units, 3)
    ])

    criterion = SoftmaxCrossEntropy()
    optimizer = SGD(model.layers, lr=learning_rate)

    loss_history = []
    progress_bar = st.progress(0)

    for epoch in range(epochs + 1):
        logits = model.forward(X)
        loss = criterion.forward(logits, y)
        
        loss_grad = criterion.backward()
        model.backward(loss_grad)
        optimizer.step()

        loss_history.append(loss)
        if epoch % (epochs // 10) == 0:
            progress_bar.progress(epoch / epochs)

    st.success("Model Training Complete!")

    # Plotting Decision Boundary
    st.subheader("Learned Decision Boundary")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    logits = model.forward(mesh_data)
    preds = np.argmax(logits, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, preds, alpha=0.8, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    st.pyplot(fig)