import torch

def generate_image_embeddings(img, model, preprocess, device):
    img = preprocess(img).unsqueeze(0).to(device) 
    with torch.no_grad():
        image_embeddings = model.encode_image(img).float()
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings.cpu().to(torch.float32).numpy()[0]
