import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load("ViT-B/32", device=device)

def generate_text_embeddings(text):
    text_features = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_features).cpu().numpy()[0]
        return text_embeddings

def generate_image_embeddings(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = model.encode_image(img_preprocessed).float()
    print(image_embeddings.shape)
    return image_embeddings[0].cpu()
    
    # plt.imshow(img)
    # plt.show()

e = generate_image_embeddings('https://m.media-amazon.com/images/I/612JNfob9nL._AC_UY218_.jpg')
print(e)