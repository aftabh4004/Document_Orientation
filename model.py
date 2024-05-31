from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn


class DocumentOrientationModel(nn.Module):
    def __init__(self, num_classes):
        super(DocumentOrientationModel, self).__init__()
        self.num_classes = num_classes
        self.vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg(x)
        return x
