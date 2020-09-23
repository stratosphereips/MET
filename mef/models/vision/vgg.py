import torch
import torch.nn as nn
import torchvision

VGG_TYPES = {"vgg_11": torchvision.models.vgg11,
             "vgg_11_bn": torchvision.models.vgg11_bn,
             "vgg_13": torchvision.models.vgg13,
             "vgg_13_bn": torchvision.models.vgg13_bn,
             "vgg_16": torchvision.models.vgg16,
             "vgg_16_bn": torchvision.models.vgg16_bn,
             "vgg_19_bn": torchvision.models.vgg19_bn,
             "vgg_19": torchvision.models.vgg19}


class Vgg(nn.Module):
    """
    Vgg model architecture using pytorch pretrained models with modifiable
    input size.
    """

    def __init__(self, vgg_type, input_dimensions, num_classes):
        super().__init__()

        if vgg_type not in VGG_TYPES:
            raise ValueError("Unknown vgg_type '{}'".format(vgg_type))

        # Load convolutional part of vgg
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=True)
        self._features = vgg.features

        # Init fully connected part of vgg
        test_input = torch.zeros(1, input_dimensions[0], input_dimensions[1], input_dimensions[2])
        test_out = vgg.features(test_input)
        n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        if vgg.classifier[0].in_features != n_features:
            self._classifier = nn.Sequential(nn.Linear(n_features, 4096),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Linear(4096, 4096),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Linear(4096, num_classes)
                                             )
            self._init_classifier_weights()
        else:
            self._classifier = vgg.classifier
            self._classifier[6].out_features = num_classes

        self._freeze_features_weights()

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self._classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(std=0.01)
                m.bias.data.zero_()

    def _freeze_features_weights(self):
        for param in self._features.parameters():
            param.requires_grad = False
