from all_clip import load_clip

def get_model():
    model, preprocess, tokenizer = load_clip(clip_model='open_clip:ViT-B-16')
    return model, preprocess, tokenizer