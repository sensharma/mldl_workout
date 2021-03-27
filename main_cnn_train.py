import os
from datetime import datetime
import pathlib

import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

from tests.cnn_dataloader import get_loaders_cifar10
from tests.cnn_model import LeNet5

MODELS_FOLDER = './models/'
SAVE_MODEL = True

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')    

    trainloader, testloader, classes = get_loaders_cifar10(batch_size=16)

    model = LeNet5()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        pbar = tqdm(trainloader, 0)
        for i, data in enumerate(pbar):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            pbar.set_description(f'Training loss: {loss.item()}')


    if SAVE_MODEL:
        now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f'{model.name}_{now_time}.pth'
        save_path = os.path.join(MODELS_FOLDER, filename)
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    print('Finished Training')
