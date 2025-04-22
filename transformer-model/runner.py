import os
import torch
import unittest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from lightning.pytorch.loggers import CSVLogger
from crohme_dataset import HMEDataModule
from transformer_model import HandwritingRecognitionModel
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOAD_EXISTING_MODEL = False
LOAD_EXISTING_CHECKPOINT = False
MODEL_PATH = None
CHECKPOINT_PATH = None


class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        # Optimize matmul precision
        torch.set_float32_matmul_precision("high")

        # Prepare data module with standard config
        self.datamodule = HMEDataModule(
            "./CACHED_CROHME/",
            batch_size=8,
            num_workers=4,
            use_pin_memory=True,
        )
        self.datamodule.setup(stage="test")

        # Initialize model architecture
        self.model = HandwritingRecognitionModel(self.datamodule.tokenizer.vocab_size, 256, 8, 1024, 0.3, 3)

        # Load from disk or train from scratch
        if LOAD_EXISTING_MODEL and os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH))
            print(f"Loaded model from {MODEL_PATH}")
        else:
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=4, verbose=True),
                ModelSummary(max_depth=2),
            ]

            self.logger = CSVLogger("logs", name="recognizer")

            # Initialize the trainer
            trainer = Trainer(
                max_epochs=60,
                callbacks=callbacks,
                accelerator="auto",
                devices="auto",
                logger=self.logger,
            )

            # If a checkpoint path is provided, load the model from the checkpoint
            if LOAD_EXISTING_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
                trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=CHECKPOINT_PATH)
                print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
            # Otherwise, train the model from scratch
            else:
                trainer.fit(self.model, datamodule=self.datamodule)

            # Save the model state dict to disk (so we can load it for testing later)
            torch.save(self.model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

        self.model.eval()


    def test_model(self):
        max_batches = None  # Set to None to run on all batches
        curr_batch = 0

        smooth = SmoothingFunction().method4

        total_bleu_greedy = 0
        total_exact_greedy = 0
        total_samples = 0

        results = []

        # Normalize function to remove spaces and convert to lowercase
        # This is used to compare the ground truth and predicted strings
        def normalize(s):
            return s.replace(" ", "").lower()

        with torch.no_grad():
            for batch in tqdm(self.datamodule.test_dataloader(), desc="Testing", unit="batch"):
                # Unpack the batch
                src, tgt, attn_mask = batch
                
                if max_batches and curr_batch >= max_batches:
                    break

                # Move to GPU if available
                src = src.to(src.device)
                tgt = tgt.to(src.device)

                # Perform greedy search
                greedy_out = self.model.greedy_search(src, self.datamodule.tokenizer)
                # Decode the target sequences
                ground_truth = [self.datamodule.tokenizer.decode(seq.tolist()) for seq in tgt]

                for truth, greedy in zip(ground_truth, greedy_out):
                    # Normalize the truth and greedy strings
                    ref = [truth.split()]
                    greedy_tokens = greedy.split()

                    # Calculate BLEU score
                    # Use the SmoothingFunction to avoid zero BLEU scores for short sentences
                    bleu_greedy = sentence_bleu(ref, greedy_tokens, smoothing_function=smooth)

                    total_bleu_greedy += bleu_greedy

                    if normalize(truth) == normalize(greedy):
                        total_exact_greedy += 1

                    results.append((truth, greedy, bleu_greedy))
                    total_samples += 1

                curr_batch += 1

        results = {
            "samples": results,
            "avg_bleu_greedy": total_bleu_greedy / total_samples,
            "exact_match_rate_greedy": total_exact_greedy / total_samples,
        }

        # Print metrics
        print(f"Avg BLEU (greedy): {results['avg_bleu_greedy']:.4f}")
        print(f"Exact Match Rate (greedy): {results['exact_match_rate_greedy']:.2%}")

        # Inspect a few results
        for i, (truth, greedy, bleu_g) in enumerate(results["samples"][:5]):
            print(f"\nExample {i+1}")
            print("GT     :", truth)
            print("Greedy :", greedy)
            print(f"BLEU (greedy): {bleu_g:.2f}")

        # Save results to file
        with open("results_test.txt", "w") as f:
            for truth, greedy, bleu_g in results["samples"]:
                f.write(f"GT: {truth}\nGreedy: {greedy}\nBLEU (greedy): {bleu_g:.2f}\n\n")

        # Save metrics to file
        with open("metrics_test.txt", "w") as f:
            f.write(f"Avg BLEU (greedy): {results['avg_bleu_greedy']:.4f}\n")
            f.write(f"Exact Match Rate (greedy): {results['exact_match_rate_greedy']:.2%}\n")

        
if __name__ == "__main__":
  unittest.main(argv=['first-arg-is-ignored'], verbosity=2)