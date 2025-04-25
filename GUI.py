import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2


try:
    model = tf.keras.models.load_model('casting_defect_detector_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def load_and_preprocess_image(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return None

def predict_image(image_path):
    processed_image = load_and_preprocess_image(image_path)
    if processed_image is not None:
        prediction = model.predict(processed_image)[0][0]
        if prediction > 0.5:
            return "Okay", prediction
        else:
            return "Defective", prediction
    else:
        return "Error", 0.0

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            prediction_result.config(text="") 
            global current_image_path
            current_image_path = file_path
        except Exception as e:
            print(f"Error loading image for display: {e}")
            prediction_result.config(text=f"Error loading image.")
            current_image_path = None

def predict_button_clicked():
    if current_image_path:
        result, probability = predict_image(current_image_path)
        if result != "Error":
            prediction_result.config(text=f"Prediction: {result} (Probability: {probability:.2f})")
    else:
        prediction_result.config(text="Please upload an image first.")


window = tk.Tk()
window.title("Casting Defect Detector")

upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

predict_button = tk.Button(window, text="Predict", command=predict_button_clicked)
predict_button.pack(pady=10)

prediction_result = tk.Label(window, text="")
prediction_result.pack()

current_image_path = None

window.mainloop()