from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CROHMEDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)

        #Take the labels of symbols and letters that can show up in the handwritten equations
        #Taken from CROHME and sort by name
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)} #has the folder names like 'x' or '1'
        
        #makes a list of image paths, name, index for all the images
        #these are all necessary for pyTorch (so we can use DataLoader)
        self.samples = [
            (img_path, label, self.class_to_idx[label])
            for label in self.classes
            for img_path in (self.root / label).glob("*.png")
        ]
        
        #use the transforms 
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2), #the is the prerocessing using pyTorch
            transforms.Resize((64, 64)),
            transforms.ToTensor()
            ])

        self.labels = self.classes

    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, idx):
        img_path, label, label_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label_idx, label