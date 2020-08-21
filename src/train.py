import os 
import pandas as pd 
import numpy as np
import albumentations
import torch
from sklearn import metrics

import dataset
import engine 
from model import get_model

if __name__ == "__main__":
    data_path = "../input/images/"
    device = "cuda"
    epochs = 10
    df = pd.read_csv("../input/train_folds.csv")
    # fetch all image ids
    images = df.image.values.tolist()
    # a list with image locations
    images = [
        os.path.join(data_path, i + ".png") for i in images
    ]
    targets = df.target.values
    model = get_model(pretrained=True)
    model.to(device)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    for fold in range(5):
        train_images = df[df.kfold!=fold].image.values.tolist()
        train_targets = df[df.kfold!=fold].target.values
        valid_images = df[df.kfold==fold].image.values.tolist()
        valid_targets = df[df.kfold==fold].target.values

        # Torch dataloader
        train_dataset = dataset.ClassificationDataset(
            image_paths=train_images,
            targets=train_targets,
            resize=(227, 227),
            augmentation=aug
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4
        )
        
        valid_dataset = dataset.ClassificationDataset(
            image_paths=valid_images,
            targets=train_targets,
            resize=(227, 227),
            augmentation=aug
        )
        valid_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=False, num_workers=4
        )

        # Simple Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

        # train and print auc score for all epochs
        for epoch in range(epochs):
            engine.train(train_loader, model, optimizer, device=device)
            predictions, valid_targets = engine.evaluate(
                valid_loader, model, device=device
            )
            roc_auc = metrics.roc_auc_score(valid_targets, predictions)
            print(
                f"Epoch={epoch}, Valid ROC AUC={roc_auc}"
            )