import torch
import math
from torch import nn, optim
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision.models import densenet121, DenseNet121_Weights
from positional_encoding import ImgPosEnc, WordPosEnc
from transformer_decoder import TransformerDecoder, TransformerDecoderLayer


# Module to rearrange tensor dimensions
class Rearranger(nn.Module):
    def __init__(self, *order):
        super().__init__()
        self.order = order

    def forward(self, tensor):
        return tensor.permute(*self.order)


# Full encoder-decoder architecture for image-to-sequence modeling
class HandwritingRecognitionModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        attention_heads: int,
        ff_hidden: int,
        drop_rate: float,
        num_layers: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Sample inputs to help Lightning understand the model
        self.example_input_array = (
            torch.rand(16, 3, 384, 512),
            torch.ones(16, 64, dtype=torch.long),
            torch.zeros(64, 64),
        )

        # CNN encoder pipeline for image feature extraction
        cnn_backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Extract the encoder layers from the DenseNet model
        # and remove the classifier layer
        encoder_layers = list(cnn_backbone.children())[:-1]
        
        # Sequentially stack the encoder layers
        seq_layer = nn.Sequential(*encoder_layers)
        # 1x1 convolutional layer to adjust the output channels
        conv_layer = nn.Conv2d(1024, d_model, kernel_size=1)
        # Rearranger to permute the tensor shape from (B, C, H, W) to (B, H, W, C)
        # IMPORTANT: this is necessary for the positional encoding to work correctly!!!
        dim_proj = Rearranger(0, 2, 3, 1)
        # Positional encoding for the image features
        pos_enc = ImgPosEnc(d_model, drop_rate)
        # Flatten the feature maps to a 2D tensor (B, H*W, C)
        flatten = nn.Flatten(1, 2)

        # Combine all layers into a single sequential model
        # The final output shape will be (B, H*W, d_model)
        self.image_encoder = nn.Sequential(seq_layer, conv_layer, dim_proj, pos_enc, flatten)

        # Transformer decoder pipeline for sequence generation
        # Token embedding layer for the input sequence
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Positional encoding for the token embeddings
        self.positional_embed = WordPosEnc(d_model, drop_rate)
        # Transformer decoder layer
        # Each layer consists of self-attention, cross-attention, and feedforward networks
        decoder_layer = TransformerDecoderLayer(d_model, attention_heads, ff_hidden, drop_rate)
        # Stack multiple decoder layers to form the full decoder
        self.sequence_decoder = TransformerDecoder(decoder_layer, num_layers)
        # Final linear layer to project the decoder output to the vocabulary size
        # This will output logits for each token in the vocabulary!! yippee!!!
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def token_decoder(self, context, tokens, attn_mask):
        # Token embedding and positional encoding
        embedded = self.token_embed(tokens) * math.sqrt(self.hparams.d_model)
        positional = self.positional_embed(embedded)
        # Decode the sequence using the transformer decoder
        decoded = self.sequence_decoder(tgt=positional, memory=context, tgt_mask=attn_mask, tgt_key_padding_mask=tokens.eq(0))
        return self.output_layer(decoded)

    # Main forward path: encode image -> decode with tokens
    def forward(self, img_tensor, tgt_seq, mask):
        encoded_img = self.image_encoder(img_tensor)
        return self.token_decoder(encoded_img, tgt_seq, mask)

    # Common loss logic reused for train/val steps
    def _compute_loss(self, batch):
        # Unpack the batch
        img, tgt, mask = batch
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        logits = self(img, tgt_input, mask[:-1, :-1])

        # Compute the loss
        return F.cross_entropy(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt_target.reshape(-1),
            ignore_index=0,
        )

    def training_step(self, batch, idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, idx):
        val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, sync_dist=True)
        return {"val_loss": val_loss}

    # Greedy decoding: pick most probable token at each step
    def greedy_search(self, src, tokenizer, max_len=256):
        with torch.no_grad():
            # src: [B, 3, H, W]
            B = src.size(0)
            # Initialize the sequence with the start token
            context = self.image_encoder(src).detach()
            # Initialize the sequence with the start token
            seqs = torch.ones(B, 1).long().to(src.device)
            # Create a mask to prevent attending to future tokens
            mask = torch.triu(torch.ones(max_len, max_len) * float("-inf"), diagonal=1).to(src.device)

            # Iterate through the sequence length
            for i in range(1, max_len):
                # Get the logits for the current sequence
                logits = self.token_decoder(context, seqs, mask[:i, :i])
                # Select the last token's logits and apply softmax to get probabilities
                next_token = logits[:, -1].log_softmax(-1).argmax(-1, keepdim=True)
                # Append the predicted token to the sequence
                seqs = torch.cat([seqs, next_token], dim=1)

        return [tokenizer.decode(seq.tolist()) for seq in seqs]

    # Optimizer/scheduler config
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
