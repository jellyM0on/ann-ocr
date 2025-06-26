import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(Y, A2):
    m = Y.shape[0]
    log_likelihood = -np.log(A2.T[range(m), np.argmax(Y, axis=1)] + 1e-9)
    return np.sum(log_likelihood) / m

def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate):
    m = X.shape[0]
    dZ2 = A2 - Y.T
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

def train_model(X, Y, hidden_size, epochs, learning_rate, update_func=None, training_loss=[]):
    print("[LOG] Starting training...")
    input_size = X.shape[1]
    output_size = Y.shape[1]
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(Y, A2)
        training_loss.append(loss)
        W1, b1, W2, b2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate)

        if epoch % 10 == 0:
            print(f"[LOG] Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
            if update_func:
                update_func()

    print("[LOG] Training complete.")
    return W1, b1, W2, b2

def predict(input_vector, W1, b1, W2, b2, index_to_ascii):
    print("[LOG] Predicting character...")
    X = input_vector.reshape(-1, 1)
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    print("[LOG] Prediction complete.")
    return index_to_ascii[np.argmax(A2)]