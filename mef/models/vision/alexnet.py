import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

from mef.models.base import Base


class AlexNet(Base):
    """
    AlexNet model architecture using pytorch pretrained models with modifiable
    input size.
    """

    def __init__(self, sample_dimensions, num_classes, model_details):
        super().__init__(sample_dimensions, num_classes, model_details)

        # Load convolutional part of resnet
        alexnet = torchvision.models.alexnet(pretrained=True)
        self._features = alexnet.features

        # Init fully connected part of resnet
        test_input = Variable(torch.zeros(1, sample_dimensions[0],
                                          sample_dimensions[1],
                                          sample_dimensions[2]))
        test_out = alexnet.features(test_input)
        n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)

        if alexnet.classifier[1].in_features != n_features:
            self._classifier = nn.Sequential(nn.Dropout(),
                                             nn.Linear(
                                                 in_features=n_features,
                                                 out_features=4096),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(),
                                             nn.Linear(
                                                 in_features=4096,
                                                 out_features=4096),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(4096, self.num_classes)
                                             )
            self._init_classifier_weights()
        else:
            self._classifier = alexnet.classifier
            self._classifier[6].out_features = self.num_classes

        self._freeze_features_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self._classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(std=0.01)
                m.bias.data.zero_()

    def _freeze_features_weights(self):
        for param in self._features.parameters():
            param.requires_grad = False
