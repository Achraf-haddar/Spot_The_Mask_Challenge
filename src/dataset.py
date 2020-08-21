import torch 
import numpy as np
from PIL import Image
from PIL import ImageFile

# When having images without ending bits This takes care of those

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None, augmentation=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = Image.convert("RGB")
        targets = self.targets[item]
        # resize
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentation is not None:
            augmented = self.augmentation(image=image)
            image = augmented["image"]
        # Pytorch expects CHW instead of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # return tensors
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }
