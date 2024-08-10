import torch

class Embeddings: 
    def __init__(self, model, preprocess, tokenizer, device):
        self.model = model
        self. preprocess = preprocess
        self. device = device
        self.tokenizer = tokenizer
    
    def generate_image_embeddings(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device) 
        with torch.no_grad():
            image_embeddings = self.model.encode_image(img).float()
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings.cpu().to(torch.float32).numpy()[0]
    
    def generate_text_embeddings(self, query):
        text_tokens = self.tokenizer(query)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embs = text_features.cpu().to(torch.float32).numpy()
        return text_embs[0]
