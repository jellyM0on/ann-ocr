import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from model import train_model, predict
from utils import extract_input_from_excel

def run_gui(X_train, Y_train, ascii_to_index, index_to_ascii):
    training_loss = []
    W1, b1, W2, b2 = train_model(
        X_train, Y_train,
        hidden_size=16,
        epochs=1000,
        learning_rate=0.5,
        update_func=None,
        training_loss=training_loss
    )

    def browse_and_predict():
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return

        try:
            input_vector, matrix = extract_input_from_excel(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read input: {e}")
            return

        ascii_code = predict(input_vector, W1, b1, W2, b2, index_to_ascii)
        char = chr(ascii_code)
        result_label.config(text=f"Prediction: '{char}' (ASCII {ascii_code})", font=("Helvetica", 20))

        ax2.clear()
        ax2.imshow(matrix, cmap="gray_r")
        ax2.set_title("Input Character (5x7)")
        ax2.axis("off")
        canvas2.draw()

    def update_loss_plot():
        ax1.clear()
        ax1.plot(training_loss, color='blue')
        ax1.set_title("Training Results")
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        canvas1.draw()

    root = tk.Tk()
    root.title("5x7 Character Recognizer")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

    container = tk.Frame(root)
    container.pack(padx=10, pady=10, fill="both", expand=True)

    frame_left = tk.Frame(container)
    frame_left.pack(side="left", padx=10, fill="both", expand=True)

    frame_right = tk.Frame(container)
    frame_right.pack(side="right", padx=10, fill="both", expand=True)

    label_train = tk.Label(frame_left, text="Training Results", font=("Helvetica", 14, "bold"))
    label_train.pack(pady=5)

    fig1, ax1 = plt.subplots(figsize=(4, 6))
    canvas1 = FigureCanvasTkAgg(fig1, master=frame_left)
    canvas1.get_tk_widget().pack()

    update_loss_plot()

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

    root.mainloop()
