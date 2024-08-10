import gradio as gr
from ..search.handle_searches import HandleSearch

class Interface:
    def __init__(self, products_collection, images_collection, search_params, model, preprocess, tokenizer, device ):
        self.search_handler = HandleSearch(products_collection, images_collection, search_params, model, preprocess, tokenizer, device)
    
    def launch_interface(self):
        text_interface = gr.Interface(
            fn=lambda text: self.search_handler.search_by_text(text),
            inputs=gr.components.Textbox(label="Search by Text"),
            outputs=gr.components.JSON(label="Results")
        )
        
        image_interface = gr.Interface(
            fn=lambda image: self.search_handler.search_by_image(image),
            inputs=gr.components.Image(label="Search by Image"),
            outputs=gr.components.JSON(label="Results")
        )
        
        text_image_interface = gr.Interface(
            fn=lambda text, image: self.search_handler.weighted_search(text, image),
            inputs=[
                gr.components.Textbox(label='Search by Text'),
                gr.components.Image(label='Search by Image')
            ],
            outputs=gr.components.JSON(label='Results')
        )
        
        iface = gr.TabbedInterface(
            interface_list=[text_interface, image_interface, text_image_interface],
            tab_names=["Text Search", "Image Search", 'Combined_weighted_search']
        )
        
        iface.launch()