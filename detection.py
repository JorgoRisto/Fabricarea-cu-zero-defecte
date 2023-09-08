import numpy as np
import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.applications import VGG16
import pickle
from ultralytics import YOLO
from collections import Counter
import os
import tempfile

with open('SVM.pkl', 'rb') as file:
    SVM = pickle.load(file)

weights = "segmentation_improved.pt"
final_model = SVM

segment_model = YOLO(weights)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

code = {0: 'crazing', 1: 'inclusion', 2: 'patches', 3: 'pitted_surface', 4: 'rolled-in_scale', 5: 'scratches'}



def load_images():
    global file_paths
    global classifications, processed_images
    current_image_index = 0
    # Reset previous data
    classifications.clear()
    processed_images.clear()
    label_aggregated_results.config(text="")
    file_paths = filedialog.askopenfilenames()

    if not file_paths:
        return

    for file_path in file_paths:
        deployment_function(file_path, root, row=3, col=4)
    show_aggregated_results()

classifications = []
processed_images = []  # To store paths of processed images

from PIL import ImageOps
# Adjusting the deployment_function to not set a default value for the window parameter


def deployment_function(image_path, window, row, col):
    global classifications
    global processed_images

    image = cv2.imread(image_path)
    model_input_image = cv2.resize(image, (200, 200))
    normalized_image = model_input_image / 255.0
    normalized_image = normalized_image.reshape((1, 200, 200, 3))
    
    # Assuming base_model and final_model are defined globally or elsewhere in your code
    features = base_model.predict(normalized_image)
    pred = np.reshape(features, (features.shape[0], -1))
    final_pred = final_model.predict(pred)
    predicted_class = code[int(final_pred)]
    
    classifications.append(predicted_class)
    
    if predicted_class not in ['crazing', 'pitted_surface']:
        results = segment_model(image, conf=0.1)
        image_to_plot = results[0].plot(masks=True, boxes=False)
    else:
        image_to_plot = image

    image_to_plot = cv2.cvtColor(image_to_plot, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_to_plot)
    
    photo_segmented = ImageTk.PhotoImage(image_pil)
    
    processed_img_path = os.path.join(tempfile.gettempdir(), os.path.basename(image_path))
    cv2.imwrite(processed_img_path, cv2.cvtColor(image_to_plot, cv2.COLOR_RGB2BGR))
    processed_images.append(processed_img_path)

    label_image_segmented = tk.Label(window)
    label_image_segmented.config(image=photo_segmented)
    label_image_segmented.photo = photo_segmented

    if window != root:
        defect_label = tk.Label(window, text=predicted_class, font=("Arial", 10, "bold"))
        defect_label.grid(row=row*2, column=2*col, padx=10, pady=5)
        
        label_image_segmented.grid(row=row*2 + 1, column=2*col, padx=10, pady=5)

    


def show_aggregated_results():
    global classifications
    total_images = len(classifications)
    
    # If no images have been processed, don't try to display results
    if total_images == 0:
        return

    class_counts = Counter(classifications)
    results_text = []
    
    for defect_type, count in class_counts.items():
        percentage = (count / total_images) * 100
        results_text.append(f"{defect_type} - {percentage:.2f}% ({count} images)")
        
    label_aggregated_results.config(text="\n".join(results_text))


current_image_index = 0

def view_images():
    global current_image_index, processed_images, classifications

    # Define constants for the grid dimensions
    ROWS = 3
    COLUMNS = 4
    NUM_IMAGES = ROWS * COLUMNS

    def display_page(start_index):
        # Clear any existing labels in the window, but spare the buttons
        for widget in image_window.winfo_children():
            if isinstance(widget, tk.Label):
                widget.destroy()

        # Display images starting from the given index
        for i in range(ROWS):
            for j in range(COLUMNS):
                idx = start_index + i * COLUMNS + j
                # Only display images if the idx is within the range of processed_images
                if idx < len(processed_images):
                    image_path = processed_images[idx]

                    # Load the image using OpenCV, then convert to the Pillow format for tkinter
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((100, 100))  # Resize or adjust as needed
                    photo = ImageTk.PhotoImage(img)

                    # Create a label with the image
                    image_label = tk.Label(image_window, image=photo)
                    image_label.image = photo  # Keep a reference to prevent garbage collection
                    image_label.grid(row=i*2, column=j, padx=10, pady=10)

                    # Display the class name below the image
                    class_label = tk.Label(image_window, text=classifications[idx], font=("Arial", 10))
                    class_label.grid(row=i*2 + 1, column=j)

        # Update button states based on images left
        # Disable 'Next' button if there are no more images to show
        if current_image_index + NUM_IMAGES >= len(processed_images):
            btn_next.config(state=tk.DISABLED)
        else:
            btn_next.config(state=tk.NORMAL)

        # Disable 'Previous' button if you're on the first page
        if current_image_index == 0:
            btn_prev.config(state=tk.DISABLED)
        else:
            btn_prev.config(state=tk.NORMAL)

    def next_page():
        global current_image_index
        if current_image_index + NUM_IMAGES < len(processed_images):
            current_image_index += NUM_IMAGES
            display_page(current_image_index)

    def previous_page():
        global current_image_index
        current_image_index = max(current_image_index - NUM_IMAGES, 0)
        display_page(current_image_index)

    # Create the new window
    image_window = tk.Toplevel(root)
    image_window.title("View Images")

    # Navigation buttons
    btn_prev = tk.Button(image_window, text="<< Previous", command=previous_page)
    btn_prev.grid(row=2*ROWS+1, column=0, columnspan=COLUMNS//2, padx=10, pady=10, sticky="ew")

    btn_next = tk.Button(image_window, text="Next >>", command=next_page)
    btn_next.grid(row=2*ROWS+1, column=COLUMNS//2, columnspan=COLUMNS//2, padx=10, pady=10, sticky="ew")

    # Display the first page by default
    display_page(current_image_index)

    # Center the window on the screen
    image_window.update_idletasks()  # Update the window to get accurate dimensions
    x = (image_window.winfo_screenwidth() - image_window.winfo_width()) // 2
    y = (image_window.winfo_screenheight() - image_window.winfo_height()) // 2
    image_window.geometry(f"+{x}+{y}")


root = tk.Tk()
root.title("Defect Detection")


# Configure the column and row weights. This will allow the grid cells to expand proportionally
# when the window is resized. 
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
for i in range(4):  # Assuming you have 4 rows
    root.grid_rowconfigure(i, weight=1)

# Increase the font size of the label
big_font = ('Arial', 24)

label = tk.Label(root, text="Defect Detection", font=big_font)
label.grid(row=0, column=0, columnspan=2)

button = tk.Button(root, text="Load Images", command=load_images)
button.grid(row=1, column=0, columnspan=2, pady=10)

#label_image_original = tk.Label(root)
#label_image_original.grid(row=2, column=0, pady=20)

label_image_segmented = tk.Label(root)
label_image_segmented.grid(row=2, column=1, pady=20)

label_class = tk.Label(root, text="")
label_class.grid(row=3, column=0, pady=20)

#label_original_text = tk.Label(root, text="")
#label_original_text.grid(row=3, column=1, pady=20)

# Label to display aggregated results
label_aggregated_results = tk.Label(root, text="", font=("Arial", 12))
label_aggregated_results.grid(row=4, column=0, columnspan=2, pady=10)

# Button to trigger the view_images function
btn_view_images = tk.Button(root, text="View Images", command=view_images)
btn_view_images.grid(row=5, column=0, columnspan=2, pady=10)

show_aggregated_results()  # Call to display aggregated results

root.mainloop()
