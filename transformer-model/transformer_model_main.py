import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision.models import densenet121
from positional_encoding import PositionalEncoding1D, PositionalEncoding2D

class TransformerEquationModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CNN Encoder (feature extractor)
        base_cnn = densenet121(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(base_cnn.features.children()))
        self.projector = nn.Conv2d(1024, d_model, kernel_size=1)

        # Positional Encoding for CNN output
        self.position_encoder_2d = PositionalEncoding2D(d_model, height=12, width=16, dropout=dropout)  # tune h/w

        # Target Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_encoder_1d = PositionalEncoding1D(d_model, dropout=dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.generator = nn.Linear(d_model, vocab_size)

    def encode(self, x):
        x = self.feature_extractor(x)  # (B, 1024, H, W)
        x = self.projector(x)         # (B, d_model, H, W)
        x = x.permute(0, 2, 3, 1)     # (B, H, W, d_model)
        x = self.position_encoder_2d(x)
        x = x.flatten(1, 2)           # (B, H*W, d_model)
        return x

    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.embedding(tgt) * (self.hparams.d_model ** 0.5)
        tgt_emb = self.position_encoder_1d(tgt_emb)
        tgt_padding_mask = tgt.eq(0)
        return self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

    def forward(self, src, tgt, tgt_mask):
        memory = self.encode(src)
        decoded = self.decode(tgt, memory, tgt_mask)
        return self.generator(decoded)

    def training_step(self, batch, batch_idx):
        src, tgt, tgt_mask = batch
        logits = self(src, tgt[:, :-1], tgt_mask[:-1, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt[:, 1:].reshape(-1),
            ignore_index=0
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, tgt_mask = batch
        logits = self(src, tgt[:, :-1], tgt_mask[:-1, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt[:, 1:].reshape(-1),
            ignore_index=0
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def beam_search(self, src, sos_token, eos_token, max_seq_len=256, beam_width=5, length_penalty=0.6):
        device = src.device
        memory = self.encode(src)  # (B, S, D)
        B = memory.size(0)
        assert B == 1, "Beam search only supports batch size 1 for now."

        # Beam state: (score, sequence)
        beams = [(0.0, [sos_token])]

        for _ in range(max_seq_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == eos_token:
                    candidates.append((score, seq))
                    continue

                tgt = torch.tensor(seq, device=device).unsqueeze(0)  # (1, T)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

                logits = self.decode(tgt, memory, tgt_mask)
                probs = F.log_softmax(self.generator(logits[:, -1]), dim=-1)  # (1, vocab)

                topk_probs, topk_indices = probs[0].topk(beam_width)
                for next_prob, next_token in zip(topk_probs, topk_indices):
                    next_seq = seq + [next_token.item()]
                    length_norm = ((5 + len(next_seq)) / 6) ** length_penalty
                    new_score = (score + next_prob.item()) / length_norm
                    candidates.append((new_score, next_seq))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            # Early stop if all beams ended
            if all(seq[-1] == eos_token for _, seq in beams):
                break

        # Return the best sequence (without SOS)
        return beams[0][1][1:]

