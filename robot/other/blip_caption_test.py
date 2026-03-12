import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tkinter as tk
from tkinter import filedialog

model_path = "./blip-image-captioning-base"

print("Loading model locally...")

processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded successfully!")
print("Using device:", device)


root = tk.Tk()
root.withdraw()  

file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("No file selected.")
    exit()

image = Image.open(file_path).convert("RGB")


inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=70,
        min_length=20,
        num_beams=5,
        repetition_penalty=1.2
    )

caption = processor.decode(output[0], skip_special_tokens=True)

print("\n========== Generated Caption ==========")
print(caption)
print("=======================================\n")