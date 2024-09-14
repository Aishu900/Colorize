import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import os.path

# Model paths
prototxt = r'model/colorization_deploy_v2.prototxt'
model = r'model/colorization_release_v2.caffemodel'
points = r'model/pts_in_hull.npy'
points = os.path.join(os.path.dirname(__file__), points)
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
model = os.path.join(os.path.dirname(__file__), model)

# Load model
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# Model parameters
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_filename=None, cv2_frame=None):
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return image, colorized

class AishwaryaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aishwarya-app")
        self.root.geometry("640x340")
        self.root.resizable(False, False)
        self.create_widgets()
    
    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)
        
        self.select_button = Button(frame, text="Select B/W image file", command=self.select_file)
        self.select_button.grid(row=0, column=0, padx=5)
        
        self.colorize_button = Button(frame, text="Colorize", command=self.colorize)
        self.colorize_button.grid(row=0, column=1, padx=5)
        
        self.original_image_panel = Label(self.root, width=300, height=300, bg='gray')
        self.original_image_panel.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.colorized_image_panel = Label(self.root, width=300, height=300, bg='gray')
        self.colorized_image_panel.pack(side=tk.LEFT, padx=10, pady=10)
    
    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if self.file_path:
            self.display_image(self.file_path, panel=self.original_image_panel)
            self.colorized_image_panel.config(image='')

    def display_image(self, file_path, panel):
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        panel.config(image=photo)
        panel.image = photo
    
    def colorize(self):
        if hasattr(self, 'file_path'):
            _, colorized_image = colorize_image(image_filename=self.file_path)
            colorized_image_pil = Image.fromarray(cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))
            colorized_image_pil.thumbnail((300, 300))
            colorized_photo = ImageTk.PhotoImage(colorized_image_pil)
            self.colorized_image_panel.config(image=colorized_photo)
            self.colorized_image_panel.image = colorized_photo

# main window
root = tk.Tk()
app = AishwaryaApp(root)
root.mainloop()
