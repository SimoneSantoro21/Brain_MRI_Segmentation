import torch
import torch.nn as nn
import wandb
import os

from libs.dataset import FLAIRDataset
from libs.U_Net import UNet
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from libs.focal import FocalLoss



wandb.init(
    # set the wandb project where this run will be logged
    project="bayesian_unet",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 5e-3,
    "architecture": "Convolutional U-Net",
    "Dropout": 0.1,
    "dataset": "FLAIR dataset",
    "epochs": 100,
    "batch size": 4,
    "Loss function": "Focal",
    "Alpha": 0.25,
    "gamma": 2,
    "Reduction": "Mean",
    "Optimizer": "AdamW",
    "Layers": 5
    }
)


def train():
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 4
    EPOCHS = 100
    DATA_PATH = "dataset"
    TRAINING_PATH = os.path.join(DATA_PATH, "training")
    VALIDATION_PATH = os.path.join(DATA_PATH, "validation")

    MODEL_SAVE_PATH = "model/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = FLAIRDataset(TRAINING_PATH)
    val_dataset = FLAIRDataset(VALIDATION_PATH)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)

    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = FocalLoss(alpha=0.25, gamma=2, reduction="mean")

    for epoch in range(EPOCHS):
        print(f"{'-' * 50}\nEpoch {epoch+1}/{EPOCHS}")

        model.train()
        train_running_loss = 0

        with tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False) as pbar:
            for index, (img, mask) in enumerate(train_dataloader):
                img = img.float().to(device)
                mask = mask.float().to(device)

                prediction = model(img)
                optimizer.zero_grad()

                loss = loss_function(prediction, mask)
                train_running_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.update(1)

        train_loss = train_running_loss / (index + 1)
        print(f"Train loss: {train_loss:.4f}")

        model.eval()
        val_running_loss = 0

        with tqdm(total=len(val_dataloader), desc=f"Validation Epoch {epoch+1}/{EPOCHS}", leave=False) as pbar:
            with torch.no_grad():
                for index, (img, mask) in enumerate(val_dataloader):
                    img = img.float().to(device)
                    mask = mask.float().to(device)

                    y_pred = model(img)
                    loss = loss_function(y_pred, mask)

                    val_running_loss += loss.item()
                    pbar.update(1)
        val_loss = val_running_loss / (index + 1)
        print(f"Validation loss: {val_loss:.4f}")

        wandb.log({"Training loss": train_loss, "Validation loss": val_loss})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return None



if __name__ == '__main__':
    train()
