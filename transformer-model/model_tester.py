import unittest
import torch
from transformer_model_main import TransformerEquationModel
from crohme_dataset import CROHMEDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_summary import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu

class TestModelTrainer(unittest.TestCase):
  def setUp(self):
    # Initialize the data module and model
    self.datamodule = CROHMEDataModule(
      "./CACHED_CROHME/",
      batch_size=8,
      num_workers=4
    )
    self.datamodule.setup(stage="fit")
    
    self.model = TransformerEquationModel(
      vocab_size=self.datamodule.tokenizer.vocab_size,
      d_model=256,
      num_heads=8,
      num_layers=4,
      ff_dim=1024,
      dropout=0.2,
      lr=1e-4
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=6, verbose=True)
    model_summary = ModelSummary(max_depth=2)

    print("Model class:", type(self.model))
    print("Module file:", self.model.__class__.__module__)

    # Initialize the trainer
    trainer = pl.Trainer(
      max_epochs=-1,
      callbacks=[early_stopping, model_summary],
      accelerator="gpu",
      devices=1
    )
    trainer.fit(model=self.model, datamodule=self.datamodule)

    self.model.eval()

    

  def test_model_beam_search(self):
    # Get a batch of validation data
    src, tgt, tgt_mask = next(iter(self.datamodule.val_dataloader()))
    src = src.to(self.model.device)
    tgt = tgt.to(self.model.device)
    tgt_mask = tgt_mask.to(self.model.device)
    tgt = tgt[:, :-1]
    tgt_mask = tgt_mask[:-1, :-1]

    # Perform beam search
    beam_preds = self.model.beam_search(src, self.datamodule.tokenizer, max_seq_len=256)
    tgt_preds = [self.datamodule.tokenizer.decode(t.tolist()) for t in tgt]

    # Evaluate using BLEU score

    bleu_scores = []
    for target, prediction in zip(tgt_preds, beam_preds):
      reference = [target.split()]
      candidate = prediction.split()
      bleu_scores.append(sentence_bleu(reference, candidate))

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {avg_bleu_score}")

    # Assert BLEU score is above a threshold (e.g., 0.5 for this test)
    self.assertGreater(avg_bleu_score, 0.5, "BLEU score is too low!")

if __name__ == "__main__":
  unittest.main()