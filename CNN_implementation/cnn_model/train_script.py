if __name__ == "__main__":
    from crohme_dataset import CROHMEDataset
    from train_cnn_model import CNNLit
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt

    #true for small subset 
    overfit_debug = False

    # dataset
    dataset = CROHMEDataset("data/symbols_labeled")

    if overfit_debug:
        # see if it works
        # use only 32 so that its faster debug
        small_set, _ = random_split(dataset, [32, len(dataset) - 32])
        #loader again
        train_loader = DataLoader(small_set, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(small_set, batch_size=32, num_workers=4, persistent_workers=True)
        max_epochs = 20
    else:
        # 80 / 20 for vlaidation 
        train_size = int(0.8 * len(dataset))
        train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=32, num_workers=4, persistent_workers=True)
        max_epochs = 20

    # model
    model = CNNLit(num_classes=len(dataset.labels))

    # pytorch lighting trainer
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto")

    # train model
    trainer.fit(model, train_loader, val_loader)

    # save weights
    trainer.save_checkpoint("symbol_model.ckpt")

    # plotting graphss
    min_len = min(len(model.train_losses), len(model.val_losses))
    train_losses = model.train_losses[:min_len]
    val_losses = model.val_losses[:min_len]
    epochs = list(range(1, min_len + 1))

    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_train_loss.png")
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_val_loss.png")
    plt.close()
