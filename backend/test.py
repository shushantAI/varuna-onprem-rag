from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import random
import easyocr
import time

t1=time.time()
caption_processor = BlipProcessor.from_pretrained("./blip_model",use_fast=False)
caption_model = BlipForConditionalGeneration.from_pretrained("./blip_model")
print(f"Caption model loading took {time.time()-t1} seconds")

#"/home/caio/Downloads/test.png"
#export CUDA_VISIBLE_DEVICES=""

t2=time.time()
ocr_reader = easyocr.Reader(
            ['en'], 
            model_storage_directory="./models",
            download_enabled=True  
        )
print(f"Ocr model loading took {time.time()-t2} seconds")

def extract_text_from_img(file_path:str) -> str:
    try:
        # Read text from image
        results = ocr_reader.readtext(file_path)
        
        # Extract text from results
        text = ' '.join([result[1] for result in results])
        
        return text
        
    except Exception as e:
        print("Error during OCR:", e)
        return ""

def generate_caption(image_path):

    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(image, return_tensors="pt")

    out_ids = caption_model.generate(**inputs)
    caption = caption_processor.decode(out_ids[0], skip_special_tokens=True)
    return caption

t3=time.time()
print(extract_text_from_img("/home/caio/Downloads/wikiVolcanoSS.png"))
print(f"Ocr  took {time.time()-t3} seconds")

t4=time.time()
print(generate_caption("/home/caio/Downloads/wikiVolcanoSS.png"))
print(f"CAPTIONING  took {time.time()-t4} seconds")
