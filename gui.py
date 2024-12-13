import customtkinter as ctk
import cv2
from train_model import train_classifier_letters, train_classifier_numbers
from recognize import run_recognition_letters, run_recognition_numbers

def open_camera_for_letters():
    run_recognition_letters()

def open_camera_for_numbers():
    run_recognition_numbers()

def train_letters():
    train_classifier_letters()
    ctk.CTkMessageBox.show_info("Training Complete", "Letters classifier training completed!")

def train_numbers():
    train_classifier_numbers()
    ctk.CTkMessageBox.show_info("Training Complete", "Numbers classifier training completed!")

# Initialize CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Sign Language Recognition")
app.geometry("800x500")

# Sidebar Frame
sidebar_frame = ctk.CTkFrame(app, width=200, corner_radius=15)
sidebar_frame.pack(side="left", fill="y", padx=10, pady=10)

sidebar_label = ctk.CTkLabel(sidebar_frame, text="Sign language", font=("Arial", 16, "bold"))
sidebar_label.pack(pady=(10, 20))

button_train_letters = ctk.CTkButton(sidebar_frame, text="Train Letters", command=train_letters)
button_train_letters.pack(pady=10)

button_train_numbers = ctk.CTkButton(sidebar_frame, text="Train Numbers", command=train_numbers)
button_train_numbers.pack(pady=10)

button_recognition_letters = ctk.CTkButton(sidebar_frame, text="Letter Recognition", command=open_camera_for_letters)
button_recognition_letters.pack(pady=10)

button_recognition_numbers = ctk.CTkButton(sidebar_frame, text="Number Recognition", command=open_camera_for_numbers)
button_recognition_numbers.pack(pady=10)

# Main Content Frame
content_frame = ctk.CTkFrame(app, corner_radius=15)
content_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

content_label = ctk.CTkLabel(content_frame, text="Sign Language Recognition", font=("Arial", 20, "bold"))
content_label.pack(pady=(20, 10))

camera_placeholder = ctk.CTkFrame(content_frame, height=300, corner_radius=15)
camera_placeholder.pack(fill="x", padx=20, pady=(10, 20))

camera_label = ctk.CTkLabel(camera_placeholder, text="Camera View Will Appear Here", font=("Arial", 16))
camera_label.pack(expand=True)

# Run the Application
app.mainloop()
