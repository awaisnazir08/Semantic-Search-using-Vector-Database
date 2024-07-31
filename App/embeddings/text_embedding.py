import torch

def get_text_query_embedding(model, tokenizer, query):
    text_tokens = tokenizer(query)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embs = text_features.cpu().to(torch.float32).numpy()
    return text_embs[0]
