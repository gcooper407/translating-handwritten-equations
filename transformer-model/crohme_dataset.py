import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
from PIL import Image
from pathlib import Path
from equation_tokenizer import EquationTokenizer


class HMEDataset(Dataset):
    def __init__(self, text_paths, image_paths, tokenizer):
        # Stores paired LaTeX and image file paths with associated preprocessing logic
        self.formula_texts = text_paths
        self.formula_images = image_paths
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.formula_texts)

    def __getitem__(self, index):
        # Load text and corresponding image at specified index
        with open(self.formula_texts[index], "r", encoding="utf-8") as file:
            formula = file.read().strip()

        # Tokenize the formula using the tokenizer
        token_ids = torch.tensor(self.tokenizer.encode(formula))

        # Load and preprocess the image
        image_tensor = transforms.ToTensor()(
          transforms.RandomPerspective(0.1, p=0.5, fill=255)(
            Image.open(self.formula_images[index]).convert("RGB")
          )
        )

        return image_tensor, token_ids


def format_batch(batch, width=512, height=384):
    # Batch contains (image_tensor, token_tensor) tuples
    imgs, sequences = zip(*batch)

    # Preprocess images to ensure they are of the same size
    encoder_input = torch.ones((len(imgs), 3, height, width))
    for idx, img in enumerate(imgs):
        h_offset = (height - img.shape[1]) // 2
        w_offset = (width - img.shape[2]) // 2
        # Center the image in the encoder input tensor
        encoder_input[idx, :, h_offset:h_offset+img.shape[1], w_offset:w_offset+img.shape[2]] = img

    # Pad the sequences to ensure they are of the same length
    decoder_input = pad_sequence(sequences, batch_first=True).long()

    # Create attention mask for the decoder input
    attn_mask = torch.triu(torch.full((decoder_input.size(1), decoder_input.size(1)), float('-inf')), diagonal=1)

    return encoder_input, decoder_input, attn_mask



class HMEDataModule(pl.LightningDataModule):
    def __init__(self, folder, batch_size=8, num_workers=4, use_pin_memory=False):
        super().__init__()
        self.path = Path(folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = use_pin_memory

    # Setup method to initialize datasets and tokenizer
    def setup(self, stage=None):
        train_txt, train_img = self.extract_data("train")
        val_txt, val_img = self.extract_data("val")
        test_txt, test_img = self.extract_data("test")

        # Obviously we want to train the tokenizer on the training dataset
        self.tokenizer = EquationTokenizer()
        equations = []
        for txt in train_txt:
            with open(txt, "r", encoding="utf-8") as file:
                equations.append(file.read().strip())

        self.tokenizer.train(equations)

        # If the stage is set to fit (or not specified), that means we're training --> need train/val sets
        if stage in ("fit", None):
            self.train_set = HMEDataset(train_txt, train_img, self.tokenizer)
            self.val_set = HMEDataset(val_txt, val_img, self.tokenizer)
            print(f"[Dataset] Training size: {len(self.train_set)}")
            print(f"[Dataset] Validation size: {len(self.val_set)}")
        # If the stage is set to test (or not specified), that means we're testing --> need test set
        if stage in ("test", None):
            self.test_set = HMEDataset(test_txt, test_img, self.tokenizer)
            print(f"[Dataset] Test size: {len(self.test_set)}")

    def extract_data(self, folder):
        # Load file paths for specified dataset
        txt_data = sorted((self.path / f"TXT/{folder}").glob("*.txt"))
        img_data = sorted((self.path / f"IMG/{folder}").glob("*.png"))

        return txt_data, img_data

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=format_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=format_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=format_batch,
        )
