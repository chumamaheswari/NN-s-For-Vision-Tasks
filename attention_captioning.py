from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load a ViT-based captioning model with attention
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load an image (replace 'path_to_image.jpg' with the actual image path)
from PIL import Image
image = Image.open("sample.jpg")

# Preprocess image
inputs = processor(images=image, return_tensors="pt")
caption_ids = model.generate(**inputs)

# Decode caption
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption with Attention: {caption}")