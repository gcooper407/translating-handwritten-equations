import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
from pathlib import Path
from hf_tokenizer import EquationTokenizer
import lightning.pytorch as pl
import torch.multiprocessing
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class CROHMEDataset(data.Dataset):
  def __init__(self, tokenizer, transform, txt_files, img_files):
    super().__init__()
    self.tokenizer = tokenizer
    self.transform = transform
    self.txt_files = txt_files
    self.img_files = img_files


  def __len__(self):
    return len(self.txt_files)

  def __getitem__(self, idx):
    with open(self.txt_files[idx], "r") as f:
      equation = f.read().strip()

    image = self.transform(Image.open(self.img_files[idx]).convert("RGB"))
    tokens = torch.tensor(self.tokenizer.encode(equation), dtype=torch.long)

    return image, tokens


class CROHMEDataModule(pl.LightningDataModule):
  def __init__(self, data_dir, batch_size=32, num_workers=4, pin_memory=False):
    super().__init__()
    self.data_dir = Path(data_dir)
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.transform = transforms.Compose([
      transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
      transforms.ToTensor()
    ])

  def setup(self, stage=None):
    self.tokenizer = EquationTokenizer()

    # These need to be sorted because .glob() returns files in arbitrary order
    # and we want to ensure that the images and text files are in the same order
    # so they match up
    train_txt_files = sorted((self.data_dir / "TXT/train").glob("*.txt"))
    train_img_files = sorted((self.data_dir / "IMG/train").glob("*.png"))
    val_txt_files = sorted((self.data_dir / "TXT/val").glob("*.txt"))
    val_img_files = sorted((self.data_dir / "IMG/val").glob("*.png"))
    
    self.tokenizer.train([str(file) for file in train_txt_files])

    if stage == "fit" or not stage:
      self.train_dataset = CROHMEDataset(self.tokenizer, self.transform, train_txt_files, train_img_files)
      self.val_dataset = CROHMEDataset(self.tokenizer, self.transform, val_txt_files, val_img_files)

  def collate_fn(self, batch, canvas_size=(512, 384)):
    max_width, max_height = canvas_size
    images, sequences = zip(*batch)

    # Prepare padded images
    batch_images = torch.ones((len(images), 3, max_height, max_width))  # White background
    for i, img in enumerate(images):
      _, h, w = img.shape
      top = (max_height - h) // 2
      left = (max_width - w) // 2
      batch_images[i, :, top:top + h, left:left + w] = img

    # Prepare padded sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)

    padded_sequences_tgt = torch.triu(torch.ones(padded_sequences.size(1), padded_sequences.size(1)) * float("-inf"), diagonal=1)

    return batch_images, padded_sequences, padded_sequences_tgt

  def train_dataloader(self):
    return data.DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      collate_fn=self.collate_fn,
      pin_memory=self.pin_memory,
      shuffle=True
    )

  def val_dataloader(self):
    return data.DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      collate_fn=self.collate_fn,
      pin_memory=self.pin_memory,
      shuffle=False
    )
  

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)

    datamodule = CROHMEDataModule("./CACHED_CROHME/", batch_size=16, num_workers=0)
    datamodule.setup(stage="fit")

    tokenizer = datamodule.tokenizer

    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    src, tgt, tgt_mask = batch  # or however you handle the batch




    # for t in tgt:
    #   print(tokenizer.decode(t.tolist()))

    # plt.imshow(make_grid(src, nrow=4).permute(1, 2, 0))
    # plt.axis("off")
    # plt.show()

    