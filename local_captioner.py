import glob
import os
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Specify the directory where your images are
image_dir = "/home/project/local_images"
image_exts = ["jpg", "jpeg", "png"]  # image file extensions to search for

# Open a file to write the captions
with open("captions_local.txt", "w", encoding="utf-8") as caption_file:
    # Iterate over each image file in the directory
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Load your image
            raw_image = Image.open(img_path).convert("RGB")

            # Process the image with a text prompt
            text = "the image of"
            inputs = processor(images=raw_image, text=text, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{img_path}: {caption}\n")
            print(f"Captioned {img_path}")