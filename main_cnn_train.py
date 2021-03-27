import os
from datetime import datetime
import pathlib

import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from tqdm import tqdm

from tests.cnn_dataloader import get_loaders_cifar10, TorchVisionDataset
from tests.cnn_model import LeNet5

MODELS_FOLDER = './models/'
SAVE_MODEL = True
DATASET_NAME = 'CIFAR10'
EPOCHS = 10

def train_epoch(train_loader, model, optimizer, criterion):
    model.train()
    avg_loss = 0
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        avg_loss += train_loss
        train_loss.backward()
        optimizer.step()
    avg_loss /= len(train_loader)
    return avg_loss.item()


def val_step(val_loader, model, optimizer, criterion):
    optimizer.zero_grad() #zero the parameter gradients
    model.eval()   # Set model to evaluate mode
    with torch.no_grad():
        val_loss = 0
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels)
        val_loss /= len(val_loader)
    return val_loss.item()

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')    

    torchvision_dataset = TorchVisionDataset(dataset_name=DATASET_NAME)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    torchvision_dataset.init_train_dataset(root='./data', download=True, transform=transform)
    train_loader, val_loader = torchvision_dataset.get_train_dataloader(batch_size=64, val_split=0.1, num_workers=2)

    model = LeNet5()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    pbar = tqdm(range(EPOCHS), unit='epoch')
    for epoch in pbar:
        # train step
        train_loss = train_epoch(train_loader, model, optimizer, criterion)

        # val step
        val_loss = val_step(val_loader, model, optimizer, criterion)

        desc = 'Train loss: {:.4f}. Val loss: {:.4f}'.format(
            train_loss, val_loss
        )
        pbar.set_description(desc)


    if SAVE_MODEL:
        now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f'{model.name}_{DATASET_NAME}_{now_time}.pth'
        save_path = os.path.join(MODELS_FOLDER, filename)
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    print('Finished Training')
    