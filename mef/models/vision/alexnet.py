import torch
import torch.nn as nn
import torchvision


class AlexNet(nn.Module):
    """
    AlexNet model architecture using pytorch pretrained models with modifiable
    input size.
    """

    def __init__(self, input_dimensions, num_classes):
        super().__init__()

        # Load convolutional part of resnet
        alexnet = torchvision.models.alexnet(pretrained=True)
        self._features = alexnet.features

        # Init fully connected part of resnet
        test_input = torch.zeros(1, input_dimensions[0], input_dimensions[1], input_dimensions[2])
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
                                             nn.Linear(4096, num_classes)
                                             )
            self._init_classifier_weights()
        else:
            self._classifier = alexnet.classifier
            self._classifier[6].out_features = num_classes

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
