import torch.nn as nn
import pretrainedmodels

def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet50"](
            pretrained="imagenet"
        )
    else:
        model = pretrainedmodels.__dict__["resnet50"](
            pretrained=None
        )
    
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=1)
    )
    return model