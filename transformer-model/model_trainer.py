from lightning.pytorch.callbacks.model_summary import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from transformer_model_main import TransformerEquationModel
from crohme_dataset import CROHMEDataModule
# from log_prediction import LogPredictionSamples

import wandb

from lightning.pytorch.callbacks import Callback


class LogPredictionSamples(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:  # log samples only for the first batch of validation data
            src, tgt, tgt_mask = batch
            tokenizer = trainer.datamodule.tokenizer

            epoch = pl_module.current_epoch
            images = [wandb.Image(img) for img in src]
            targets = [tokenizer.decode(seq.tolist()) for seq in tgt]
            beams = pl_module.beam_search(src, tokenizer)

            wandb_logger.log_text(
                key="sample_latex",
                columns=["epoch", "image", "target", "beam"],
                data=[
                    [epoch, i, t, b]
                    for i, t, b in zip(images, targets, beams)
                ],
            )


# if __name__ == '__main__':

#   wandb_logger = pl.loggers.WandbLogger()

#   datamodule = CROHMEDataModule(
#       "./CACHED_CROHME/",
#       batch_size=16,
#       num_workers=0
#   )
#   datamodule.setup(stage="fit")
#   model = TransformerEquationModel(
#       vocab_size=datamodule.tokenizer.vocab_size,
#       d_model=256,
#       num_heads=8,
#       num_layers=4,
#       ff_dim=1024,
#       dropout=0.2,
#       lr=1e-4
#   )

#   early_stopping = EarlyStopping(monitor="val_loss", patience=6, verbose=True)
#   model_summary = ModelSummary(max_depth=2)
#   log_prediction_samples = LogPredictionSamples()

#   trainer = pl.Trainer(
#       max_epochs=-1,
#       logger=wandb_logger,
#       callbacks=[early_stopping, model_summary, log_prediction_samples],
#       accelerator="gpu",
#       devices=2,
#   )
#   trainer.fit(model=model, datamodule=datamodule)

#   model.eval()

#   src, tgt, tgt_mask = next(iter(datamodule.val_dataloader()))
#   src = src.to(model.device)
#   tgt = tgt.to(model.device)
#   tgt_mask = tgt_mask.to(model.device)
#   tgt = tgt[:, :-1]
#   tgt_mask = tgt_mask[:-1, :-1]
#   # logits = model(src, tgt, tgt_mask)
  

#   beam_preds = model.beam_search(src, datamodule.tokenizer, max_seq_len=256)
#   tgt_preds = [datamodule.tokenizer.decode(t.tolist()) for t in tgt]

  





  
  
  
  

