import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("brain_tumor_model.joblib")

# Class label mapping
desc = {1: "Positive Tumor", 0: "No Tumor"}

# Upload and predict function
def upload_img():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    try:
        # Load and preprocess image for prediction
        img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray, (200, 200))
        img_array = img_gray.reshape(1, -1) / 255.0

        # Display image in GUI
        img_display = Image.open(file_path)
        img_display = img_display.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_display)

        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Predict and show result
        prediction = model.predict(img_array)
        result_text = f"Prediction: {desc[prediction[0]]}"
        op.config(text=result_text, fg="green" if prediction[0] == 1 else "blue")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image:\n{e}")

# GUI setup
app = tk.Tk()
app.title("Brain Tumor Classifier")
app.geometry("800x600")
app.configure(bg="gray")

# Title Label
title = tk.Label(app, text="Brain Tumor Classifier", font=("Arial", 20, "bold"), bg="gray", fg="black")
title.pack(pady=10)

# Frame for Image
frame = tk.Frame(app, bg="white")
frame.place(x=300, y=80, width=200, height=200)

# Upload Button
btn = tk.Button(app, text="Upload MRI", font=("Arial", 15), bg="darkgray", command=upload_img)
btn.place(x=330, y=300)

# Image Display Label
img_label = tk.Label(frame)
img_label.pack()

# Output Prediction Label
op = tk.Label(app, font=("Arial", 15, "bold"), bg="gray")
op.place(x=300, y=400)

# Start GUI
app.mainloop()

