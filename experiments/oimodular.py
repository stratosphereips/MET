import torch
import torchvision.transforms as T

from mef.utils.experiment import train_victim_model
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.datasets.vision.oimodular import OIModular
from mef.utils.pytorch.models.vision import ResNet

if __name__ == "__main__":
    train_set = OIModular("E:\Datasets\OpenImagesv6", 5, download=True,
                          transform=T.Compose(
                                  [T.Grayscale(num_output_channels=3),
                                   T.Resize((224, 224)),
                                   T.ToTensor()]))
    test_set = OIModular("E:\Datasets\OpenImagesv6", 5, train=False,
                         download=True,
                         transform=T.Compose([
                             T.Grayscale(num_output_channels=3),
                             T.Resize((224, 224)),
                             T.ToTensor()]))

    train_set, val_set = split_dataset(train_set, 0.2)

    test_model = ResNet("resnet_34", 5)
    optimizer = torch.optim.Adam(test_model.parameters())
    loss = torch.nn.functional.cross_entropy

    train_victim_model(test_model, optimizer, loss, train_set,
                       5, 1000, 64, 4, val_set, gpus=1)
