from all_clip import load_clip
from ..utils.helper_utils import load_config
class Model:
    def __init__(self):
        self.config = load_config()
        self.clip_model = self.config['model_name']
    
    def load_model(self):
        model, preprocess, tokenizer = load_clip(self.clip_model)
        return model, preprocess, tokenizer


if __name__ =='__main__':
    model = Model()