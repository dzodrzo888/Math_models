import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100, noise_std=0.8):
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    true_w = 3.7
    true_b = 2.5
    y = true_w * X + true_b + np.random.randn(n_samples, 1) * noise_std
    return X, y

def predict(X, w, b):
    """
    Predicts y_hat using the lin reg equation

    Args:
        X (np.array): X
        w (float): weights
        b (float): bias

    Returns:
        np.array: Predicted y value
    """
    y_hat = w * X + b
    return y_hat

def compute_cost(y_hat, y):
    m = len(y)
    mse = np.sum(np.square(y_hat - y)) / m
    return mse

def compute_gradients(X, y, y_hat):
    dw = (2/len(y))*np.sum((y_hat - y)*X)
    db = (2/len(y))*np.sum(y_hat - y)
    return dw, db

def train(X, y, learning_rate=0.01, epochs=1000):
    w = 0.0
    b = 0.0

    losses = []
    
    for epoch in range(epochs):

        y_hat = predict(X, w, b)

        cost = compute_cost(y_hat, y)

        losses.append(cost)

        dw, db = compute_gradients(X, y, y_hat)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4}: Cost = {cost:.4f} | w = {w:.4f}, b = {b:.4f}")

    return w, b, losses

def plot_results(X, y, w, b, losses):
    # Plot regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', label='Data')
    y_pred = predict(X, w, b)
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Fitted Line")

    # Plot loss over time
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = generate_data()
    w, b, losses = train(X, y, learning_rate=0.05, epochs=1000)
    plot_results(X, y, w, b, losses)
