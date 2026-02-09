from transformers import AutoProcessor, AutoModelForImageTextToText
from visualize_data import visualize_data
from extract_data_from_hdf5 import extract_episode_data
from generate_prompt import generate_prompt
from PIL import Image
import argparse

def convert_to_pil(images):
    return [Image.fromarray(img) for img in images]

def generate_action_caption(llm_model, prompt, visualize):
    processor = AutoProcessor.from_pretrained(llm_model)
    model = AutoModelForImageTextToText.from_pretrained(llm_model)

    text, images, images_next = prompt
    
    images = convert_to_pil(images)
    images_next = convert_to_pil(images_next)

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "image", "image": images[2]},
            {"type": "image", "image": images_next[0]},
            {"type": "image", "image": images_next[1]},
            {"type": "image", "image": images_next[2]},
            {"type": "text", "text": text}
        ]
    },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=2000)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

    if visualize:
        visualize_data(text, [images, images_next])

def main():
    
    


    
        

if __name__ == "__main__":
    main()