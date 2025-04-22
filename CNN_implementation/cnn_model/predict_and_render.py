import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import json
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

from train_cnn_model import CNNLit
from hf_tokenizer import EquationTokenizer

# paths
input_dir = Path("data/symbols_labeled")
output_dir = Path("predictions")
output_dir.mkdir(exist_ok=True)
tokenizer_path = Path("tokenizer.json")

# labels
labels = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
idx_to_label = {i: label for i, label in enumerate(labels)}

# model
model = CNNLit(num_classes=len(idx_to_label))
ckpt = torch.load("symbol_model.ckpt")
model.load_state_dict(ckpt.get("state_dict", ckpt))
model.eval()

# tokenizer for equation 
tokenizer = EquationTokenizer()
if tokenizer_path.exists():
    tokenizer.load(str(tokenizer_path))
else:
    #fix the errors with the namings that inlcuded special characters 
    formula = [label.replace("_slash_", "/").replace("_mul_", "*") for label in labels] 

    with open("tmp_formula.txt", "w") as f:
        f.write("\n".join(formula))
    tokenizer.train(["tmp_formula.txt"])
    tokenizer.save(str(tokenizer_path))

# Transform images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

latex_tokens = []
pred_sequence = [] #keepign track of the predicted and expected now 
expected_sequence = []
correct = total = 0

for folder in input_dir.iterdir():
    if not folder.is_dir():
        continue
    expected = folder.name.replace("_slash_", "/").replace("_mul_", "*") #fix the namings again
    for img_path in folder.glob("*.png"):
        img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            pred = model(img).argmax(dim=1).item()
        pred_token = idx_to_label[pred].replace("_slash_", "/").replace("_mul_", "*") #fix the namings again
        correct += pred_token == expected
        total += 1
        pred_sequence.append(pred_token) 
        expected_sequence.append(expected)
        latex_tokens.append((img_path.name, pred_token, expected, pred_token == expected))

# predictions are in json format in predictions.json
with open(output_dir / "predictions.json", "w") as f:
    json.dump([{
        "filename": name,
        "token": pred,
        "expected": exp,
        "correct": ok
    } for name, pred, exp, ok in latex_tokens], f, indent=2)

# scores!!!
acc = correct / total * 100 if total > 0 else 0 
bleu = sentence_bleu([expected_sequence], pred_sequence) #use the bleu score func
print(f"Accuracy: {acc:.2f}%")
print(f"BLEU Score: {bleu:.4f}")
