from PIL import Image
import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)

def generate_blip2_captions(image, num_beams=5, num_return_sequences=3):
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", local_files_only=True)
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", local_files_only=True)
    except Exception:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences)
    return [processor.decode(o, skip_special_tokens=True) for o in outputs]

def generate_blip2_opt_captions(image, num_beams=5, num_return_sequences=3):
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b", local_files_only=True)
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", local_files_only=True)
    except Exception:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b")
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences)
    return [processor.decode(o, skip_special_tokens=True) for o in outputs]

def generate_blip_captions(image, num_beams=5, num_return_sequences=3):
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
    except Exception:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences)
    return [processor.decode(o, skip_special_tokens=True) for o in outputs]

def get_clip_best_caption(image, candidate_captions):
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    except Exception:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=candidate_captions, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_idx = torch.argmax(probs, dim=1).item()
    return candidate_captions[best_idx]

def generate_best_caption(image):
    blip2_caps = generate_blip2_captions(image, num_beams=3, num_return_sequences=2)
    blip2_opt_caps = generate_blip2_opt_captions(image, num_beams=3, num_return_sequences=2)
    blip_caps = generate_blip_captions(image, num_beams=3, num_return_sequences=2)
    candidate_captions = list(set(blip2_caps + blip2_opt_caps + blip_caps))
    best_caption = get_clip_best_caption(image, candidate_captions)
    return best_caption
