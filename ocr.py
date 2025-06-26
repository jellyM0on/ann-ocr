import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_training_data(file_path):
    print("[LOG] Loading training data...")
    df = pd.read_excel(file_path)

    print("[LOG] Extracting features and labels...")
    X = df.iloc[:, 1:36].values.astype(int)
    y_ascii = df['label'].values

    ascii_classes = sorted(set(y_ascii))
    ascii_to_index = {val: i for i, val in enumerate(ascii_classes)}
    index_to_ascii = {i: val for val, i in ascii_to_index.items()}

    Y = np.zeros((len(y_ascii), len(ascii_classes)))
    for i, val in enumerate(y_ascii):
        Y[i, ascii_to_index[val]] = 1

    print(f"[LOG] Loaded {len(X)} samples.")
    return X, Y, ascii_to_index, index_to_ascii

# Step 2: Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    print("[LOG] Initializing network parameters...")
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    print("[LOG] Initialization complete.")
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
    loss = np.sum(log_likelihood) / m
    return loss

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

training_loss = [] 

def train_model(X, Y, hidden_size=16, epochs=1000, learning_rate=0.1, update_chart_func=None):
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
            print(f"[LOG] Epoch {epoch} | Loss: {loss:.4f}")
            if update_chart_func:
                update_chart_func()

    print("[LOG] Training complete.")
    return W1, b1, W2, b2

def predict(input_vector, W1, b1, W2, b2, index_to_ascii):
    print("[LOG] Making prediction...")
    if input_vector.shape[0] != W1.shape[1]:
        print("Input vector shape:", input_vector.shape)
        print("W1 shape:", W1.shape)
        raise ValueError(f"[ERROR] Input vector has shape {input_vector.shape}, expected {W1.shape[1]} features.")

    X = input_vector.reshape(-1, 1)
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    predicted_index = np.argmax(A2)
    ascii_code = index_to_ascii[predicted_index]
    print(f"[LOG] Prediction complete: ASCII {ascii_code} ({chr(ascii_code)})")
    return ascii_code

def extract_input_from_excel(file_path):
    print("[LOG] Extracting input vector from Excel file...")
    df = pd.read_excel(file_path, header=None)
    df = df.reindex(index=range(7), columns=range(5), fill_value=0)
    matrix = (df.fillna(0).to_numpy() != 0).astype(int)

    # Flatten to shape (35,)
    input_vector = matrix.flatten()

    print(matrix)
    print(input_vector)

    print(f"[LOG] Extraction complete. Input vector shape: {input_vector.shape}")
    return input_vector

def run_gui():
    def update_loss_plot():
        ax1.clear()
        ax1.plot(training_loss, color='blue')
        ax1.set_title("Training Results")
        ax1.set_xlabel("Epochs", fontsize=10)  
        ax1.set_ylabel("Loss", fontsize=10)  
        canvas1.draw()

    def browse_and_predict():
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return

        try:
            input_vector = extract_input_from_excel(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read input: {e}")
            return

        predicted_ascii = predict(input_vector, W1, b1, W2, b2, index_to_ascii)
        char = chr(predicted_ascii)
        result_label.config(
            text=f"Prediction: '{char}' (ASCII {predicted_ascii})",
            font=("Helvetica", 20)
        )

        matrix = input_vector.reshape((7, 5))
        ax2.clear()
        ax2.imshow(matrix, cmap="gray_r")
        ax2.set_title("Input Character (5x7)")
        ax2.axis("off")
        canvas2.draw()

    root = tk.Tk()
    root.title("5x7 Character Recognizer")
    root.attributes("-fullscreen", True)

    def exit_fullscreen(event):
        root.attributes("-fullscreen", False)

    root.bind("<Escape>", exit_fullscreen)

    container = tk.Frame(root)
    container.pack(padx=10, pady=10, fill="both", expand=True)

    frame_left = tk.Frame(container)
    frame_left.pack(side="left", padx=10, fill="both", expand=True)

    frame_right = tk.Frame(container)
    frame_right.pack(side="right", padx=10, fill="both", expand=True)

    label_train = tk.Label(frame_left, text="Training Results", font=("Helvetica", 14, "bold"))
    label_train.pack(pady=5)

    fig1, ax1 = plt.subplots(figsize=(4, 8))
    canvas1 = FigureCanvasTkAgg(fig1, master=frame_left)
    canvas1.get_tk_widget().pack()

    # RIGHT: Input + Prediction
    tk.Button(frame_right, text="Select Input Excel File", command=browse_and_predict).pack(pady=10)

    fig2, ax2 = plt.subplots(figsize=(2, 3))
    ax2.axis("off")
    ax2.set_title("Input Character (5x7)")
    blank_input = np.zeros((7, 5))
    ax2.imshow(blank_input, cmap="gray_r")
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_right)
    canvas2.get_tk_widget().pack()

    result_label = tk.Label(frame_right, text="Prediction: ", font=("Helvetica", 16))
    result_label.pack(pady=10)

    root.after(100, update_loss_plot)  
    root.mainloop()

if __name__ == "__main__":
    print("[LOG] Running character recognition model...")

    training_file = "training.xlsx"
    X_train, Y_train, ascii_to_index, index_to_ascii = load_training_data(training_file)

    run_gui_training_ready = lambda: None
    W1, b1, W2, b2 = train_model(
        X_train, Y_train,
        hidden_size=16,
        epochs=1000,
        learning_rate=0.5,
        update_chart_func=run_gui_training_ready  
    )

    run_gui()
