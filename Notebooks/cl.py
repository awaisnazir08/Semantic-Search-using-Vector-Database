import torch
import clip
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Example text
text = "A description of your product"

# Preprocess and tokenize the text
text_tokens = clip.tokenize([text]).to(device)

# Compute text embeddings
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)

# Load and preprocess the image
image_path = "pic.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Compute image embeddings
with torch.no_grad():
    image_embeddings = model.encode_image(image)

# Convert embeddings to numpy arrays
text_embeddings = text_embeddings.cpu().numpy().flatten()
image_embeddings = image_embeddings.cpu().numpy().flatten()

print("Text Embeddings Shape:", text_embeddings.shape)
print("Image Embeddings Shape:", image_embeddings.shape)

print(image_embeddings)