import os
import cv2
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hf_tokenizer import EquationTokenizer
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# organize the paths 
img_folder = Path("CACHED_CROHME/IMG/CROHME2023_train")
label_folder = Path("CACHED_CROHME/TXT/CROHME2023_train")
OUTPUT_DIR = Path("data/symbols_labeled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# tokenizer for all of the label files
tokenizer = EquationTokenizer()
text_files = list(label_folder.glob("*.txt"))
tokenizer.train([str(f) for f in text_files])

# transform for cnn 
img_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

img_id = 0  # for file names will need later

# run through txt files 
for label_file in tqdm(text_files, desc="Segmenting"):
    img_file = img_folder / f"{label_file.stem}.png"
    if not img_file.exists():
        continue

    # grayscale image using opencv
    img_gray = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

    # read then tokenize str
    with open(label_file, 'r') as f:
        raw_equation = f.read().strip()
    tokens = tokenizer.tokenizer.pre_tokenizer.pre_tokenize_str(raw_equation)
    tokens = [token[0] for token in tokens]

    # More opencv processes --> image to binary  (this is required for components)
    e, binary_img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Finally apply conected components to get the symbols
    num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    boxes = [(i, stats[i]) for i in range(1, num_components)]
    boxes.sort(key=lambda b: b[1][0])  # put the left most symbol on the left and so on
    # needed to sort to put it in the right order for the equation again

    # save the predicted toekns as a png 
    for i, (_, (x, y, w, h, _)) in enumerate(boxes):
        if i >= len(tokens):
            break

        symbol_crop = img_gray[y:y+h, x:x+w] #get the exterior box lining 
        symbol_img = Image.fromarray(symbol_crop).convert("RGB") #3 channel img so that we can use easily later
        symbol_tensor = img_transform(symbol_img) #transform for pytorch again

        
        token = tokens[i].replace('/', '_slash_').replace('*', '_mul_') # fix the naming again 
        label_dir = OUTPUT_DIR / token
        label_dir.mkdir(parents=True, exist_ok=True)

        save_path = label_dir / f"{label_file.stem}_{i}_{img_id}.png"
        img_array = (symbol_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_array).save(save_path)

        img_id += 1
