import torch
import torch.nn as nn
import wandb
from data import (
    generate_mask_matrix,
    generate_data,
    read_used_pathways,
    NUM_SPLITS,
)
from model import PNet

BATCH_COUNT = 20
EPOCHS = 50
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    wandb.init(
        project="pnet-training",
        name="Exp2. Directed nodes",
        config={
            "epochs": EPOCHS,
            "learning_rate": LR,
            "hidden_dim": 8,
            "num_splits": NUM_SPLITS,
        },
    )

    Xvalid, yvalid = generate_data(False)
    Xvalid = Xvalid.to(device)
    yvalid = yvalid.to(device)

    Xsample, _ = generate_data(True, 0)
    N, gene_dim = Xsample.shape
    pathway_dim = len(read_used_pathways())
    hidden_dim = 128
    output_dim = 2
    gene_mask = generate_mask_matrix().to(device)

    print(f"Gene mask shape: {gene_mask.shape}")
    print(f"Gene dim: {gene_dim}, Pathway dim: {pathway_dim}")

    model = PNet(0, gene_dim, pathway_dim, hidden_dim, output_dim, gene_mask).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    training_losses = []
    validation_losses = []
    validation_accuracies = []

    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_losses = []

        for split in range(NUM_SPLITS):
            Xtrain, ytrain = generate_data(True, split)
            Xtrain = Xtrain.to(device)
            ytrain = ytrain.to(device)
            # Convert one-hot to class indices
            ytrain_idx = torch.argmax(ytrain, dim=1)

            optimizer.zero_grad()
            outputs = model(Xtrain)
            loss = criterion(outputs, ytrain_idx)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        model.eval()
        with torch.no_grad():
            val_outputs = model(Xvalid)
            yvalid_idx = torch.argmax(yvalid, dim=1)  # Convert one-hot
            val_loss = criterion(val_outputs, yvalid_idx)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == yvalid_idx).float().mean()

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss.item(),
                "val_acc": val_acc.item(),
            }
        )

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={val_loss.item():.4f}, Val Acc={val_acc.item():.4f}"
            )

        training_losses.append(avg_train_loss)
        validation_losses.append(val_loss.item())
        validation_accuracies.append(val_acc.item())


if __name__ == "__main__":
    train()
