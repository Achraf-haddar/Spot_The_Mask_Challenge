import torch
import torch.nn as nn
from tqdm import tqdm

def train(data_loader, model, optimzier, device):
    # put the model in train mode
    model.train()

    for data in data_loader:
        inputs = data["image"]
        targets = data["targets"]
        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = inputs.to(device, dtype=torch.float)

        # zero grad the optimzier
        optimzier.zero_grad()
        # do the forward step of model
        outputs = model(inputs)
        # calculate loss
        loss = nn.BCEWithLogitsLoss()(outputs, target.view(-1, 1))
        # backward step the loss
        loss.backward()
        # step the optimzier
        optimizer.step()

def evaluate(data_loader, model, device):
    # put the model in evaluation mode
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # Forward pass to generate prediction
            output = model(inputs)
            # convert targets and outputs to list
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            # extend the original list
            final_targets.extend(targets)
            final_outputs.extend(output)
    
    return final_outputs, final_targets