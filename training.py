import torch
#import wandb
import os
import argparse

from libs.dataset import FLAIRDataset
from libs.U_Net import UNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from libs.focal import FocalLoss



#wandb.init(
#    # set the wandb project where this run will be logged
#    project="bayesian_unet",
#
#    # track hyperparameters and run metadata
#    config={
#    "learning_rate": 5e-3,
#    "architecture": "Convolutional U-Net",
#    "Dropout": 0.1,
#    "dataset": "FLAIR dataset",
#    "epochs": 100,
#    "batch size": 4,
#    "Loss function": "Focal",
#    "Alpha": 0.25,
#    "gamma": 2,
#    "Reduction": "Mean",
#    "Optimizer": "AdamW",
#    "Layers": 5
#    }
#)

def arg_parser():
    description = 'Bayesian U-Net training'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input',
                        required=True,
                        type=str,
                        help='Path of the input data directory')
    
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help='Path to the saving location of the output model')
    
    parser.add_argument('--lr',
                        type=float,
                        default=5e-4,
                        help='Model learning rate')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of training epochs')
    
    parser.add_argument('--batch',
                        type=int,
                        default=8,
                        help='Batch size')
    
    args = parser.parse_args()
    return args
    

def train(data_path, model_save_path, learning_rate, epochs, batch_size):
    """
    Trains a Bayesian U-Net model on the provided dataset.

    Args:
        data_path (str): Path to the dataset directory containing 'training' and 'validation' subdirectories.
        model_save_path (str): Path to save the trained model.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation dataloaders.

    Returns:
        None
    """
    TRAINING_PATH = os.path.join(data_path, "training")
    VALIDATION_PATH = os.path.join(data_path, "validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = FLAIRDataset(TRAINING_PATH)
    val_dataset = FLAIRDataset(VALIDATION_PATH)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True, drop_last=True)

    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = FocalLoss(alpha=0.25, gamma=2, reduction="mean")

    for epoch in range(epochs):
        print(f"{'-' * 50}\nEpoch {epoch+1}/{epochs}")

        model.train()
        train_running_loss = 0

        with tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch+1}/{epochs}", leave=False) as pbar:
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

        with tqdm(total=len(val_dataloader), desc=f"Validation Epoch {epoch+1}/{epochs}", leave=False) as pbar:
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

        #wandb.log({"Training loss": train_loss, "Validation loss": val_loss})

    torch.save(model.state_dict(), model_save_path)

    return None



if __name__ == '__main__':
    args = arg_parser()
    train(args.input, args.output, args.lr, args.epochs, args.batch)
