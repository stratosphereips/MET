import os
import sys

import torch
import torch.nn.functional as F
from pl_bolts.models.gans import GAN
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.ripper import Ripper
from mef.attacks.base import BaseSettings, TrainerSettings
from mef.utils.pytorch.datasets.vision import ImageNet1000
from mef.utils.pytorch.models.vision import AtCnn
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets import split_dataset
from mef.utils.pytorch.lighting.module import MefModule
from mef.utils.pytorch.lighting.training import get_trainer

IMAGENET_TRAIN_SIZE = 100000
LATENT_DIM = 128
DIMS = (3, 32, 32)
NUM_CLASSES = 10


def set_up(args):
    seed_everything(args.seed)

    victim_model = AtCnn(dims=DIMS, num_classes=NUM_CLASSES,
                         dropout_keep_prob=0.2)
    substitute_model = AtCnn(dims=DIMS, num_classes=NUM_CLASSES,
                             dropout_keep_prob=0.2)
    generator = GAN(*DIMS, latent_dim=LATENT_DIM, learning_rate=0.0001)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()
        generator.cuda()

    # Prepare data
    print("Preparing data")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([T.CenterCrop(DIMS[1]), T.ToTensor(),
                           T.Normalize(mean, std)])
    train_set = CIFAR10(root=args.cifar10_dir, download=True,
                        transform=transform)
    test_set = CIFAR10(root=args.cifar10_dir, train=False, download=True,
                       transform=transform)

    transform = T.Compose([T.CenterCrop(DIMS[1]), T.ToTensor(),
                           T.Normalize(mean, std)])
    imagenet_train = ImageNet1000(root=args.imagenet_dir,
                                  size=IMAGENET_TRAIN_SIZE,
                                  transform=transform, seed=args.seed)

    try:
        saved_model = torch.load(args.save_loc +
                                 "/victim/final_victim_model.pt")
        victim_model.load_state_dict(saved_model["state_dict"])
        print("Loaded victim model")
    except FileNotFoundError:
        print("Training victim model")
        optimizer = torch.optim.Adam(victim_model.parameters(),
                                     weight_decay=1e-3)
        loss = F.cross_entropy

        train_set, val_set = split_dataset(train_set, 0.2)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=args.batch_size)

        val_dataloader = DataLoader(dataset=val_set, pin_memory=True,
                                    num_workers=4, batch_size=args.batch_size)

        mef_model = MefModule(victim_model, NUM_CLASSES, optimizer, loss)
        base_settings = BaseSettings(gpus=args.gpus, save_loc=args.save_loc)
        trainer_settings = TrainerSettings(training_epochs=1000,
                                           evaluation_frequency=1,
                                           patience=args.patience,
                                           precision=args.precision)
        trainer = get_trainer(base_settings, trainer_settings, "victim")
        trainer.fit(mef_model, train_dataloader, val_dataloader)

        torch.save(dict(state_dict=victim_model.state_dict()),
                   args.save_loc + "/victim/final_victim_model.pt")

    try:
        saved_generator = torch.load(args.save_loc +
                                     "/generator/final_generator.pt")
        generator.load_state_dict(saved_generator["state_dict"])
        print("Loaded Generator")
    except FileNotFoundError:
        print("Training Generator")

        train_dataloader = DataLoader(dataset=imagenet_train, shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      batch_size=args.batch_size)

        base_settings = BaseSettings(gpus=args.gpus, save_loc=args.save_loc)
        trainer_settings = TrainerSettings(
                training_epochs=args.substitute_train_epochs,
                patience=args.patience, precision=args.precision)
        trainer = get_trainer(base_settings, trainer_settings, "victim")
        trainer.fit(generator, train_dataloader)

        torch.save(dict(state_dict=generator.state_dict()),
                   args.save_loc + "/generator/final_victim_model.pt")

    def visualize():
        import numpy as np
        import matplotlib.pyplot as plt
        while 1:
            image = generator(
                    torch.Tensor(np.random.normal(size=(1, 128))).cuda())
            image = image.clamp(-1, 1) / 2. + .5
            image = image.detach().cpu().numpy().transpose([0, 2, 3, 1])[0]
            plt.imshow(image)
            plt.show()

    visualize()

    return victim_model, substitute_model, generator, test_set


if __name__ == "__main__":
    parser = Ripper.get_attack_args()
    parser.add_argument("--cifar10_dir", default="./data/", type=str,
                        help="Path to CIFAR10 dataset (Default: ./data/)")
    parser.add_argument("--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, generator, test_set = set_up(args)
    af = Ripper(victim_model, substitute_model, generator, LATENT_DIM,
                args.greyscale, args.generated_data, args.budget,
                args.output_type)

    # Baset settings
    af.base_settings.save_loc = args.save_loc
    af.base_settings.gpus = args.gpus
    af.base_settings.seed = args.seed
    af.base_settings.deterministic = args.deterministic
    af.base_settings.debug = args.debug

    # Trainer settings
    af.trainer_settings.training_epochs = args.substitute_train_epochs
    af.trainer_settings.patience = args.patience
    af.trainer_settings.evaluation_frequency = args.evaluation_frequency
    af.trainer_settings.precision = args.precision
    af.trainer_settings.accuracy = args.accuracy

    # Data settings
    af.data_settings.batch_size = args.batch_size

    af(test_set)
