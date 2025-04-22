import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from cnn_model import CNN  # Your custom CNN class

# pytorch to train
class CNNLit(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.model = CNN(num_classes)   # CNN model 
        self.lr = lr                # learning rate
        self.train_losses = []         #loss per epoch
        self.val_losses = []           #validation loss

    def forward(self, x):
        return self.model(x)  # forward

    def training_step(self, batch, _):
        images, labels, _ = batch
        preds = self(images)
        loss = F.cross_entropy(preds, labels)  #  cross entropy caluclation is here
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    #do same for validation 
    def validation_step(self, batch, _):
        images, labels, _ = batch
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # save training loss (use this for graph )
        latest = self.trainer.callback_metrics.get("train_loss")
        if latest is not None:
            self.train_losses.append(latest.item())

    def on_validation_epoch_end(self):
        #validation loss
        latest = self.trainer.callback_metrics.get("val_loss")
        if latest is not None:
            self.val_losses.append(latest.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)  # use Adam optimizer 
