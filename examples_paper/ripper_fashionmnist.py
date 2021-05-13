import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.utils.pytorch.models.generators import Sngan
from mef.attacks.ripper import Ripper
from mef.utils.experiment import train_victim_model
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets.vision import FashionMnist
from mef.utils.pytorch.lighting.module import Generator, TrainableModel, VictimModel
from mef.utils.pytorch.functional import soft_cross_entropy
from mef.utils.pytorch.models.vision import LeNet, HalfLeNet
from mef.utils.pytorch.blocks import ConvBlock, MaxPoolLayer

LATENT_DIM = 128
DIMS = (1, 32, 32)
NUM_CLASSES = 10


def set_up(args):
    seed_everything(args.seed)

    victim_model = LeNet(NUM_CLASSES)
    substitute_model = HalfLeNet(NUM_CLASSES)
    generator = Sngan(
        args.generator_checkpoint,
        resolution=32,
        transform=T.Compose([T.Grayscale()]),
    )

    # Prepare data
    print("Preparing data")
    transform = T.Compose([T.Pad(2), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_set = FashionMnist(
        root=args.fashion_mnist_dir, transform=transform, download=True
    )
    test_set = FashionMnist(
        root=args.fashion_mnist_dir, train=False, transform=transform, download=True
    )

    vict_training_epochs = 200
    train_victim_model(
        victim_model,
        torch.optim.Adam,
        F.cross_entropy,
        train_set,
        NUM_CLASSES,
        vict_training_epochs,
        args.batch_size,
        args.num_workers,
        test_set=test_set,
        save_loc=Path(args.save_loc).joinpath("victim"),
        gpu=args.gpu,
        deterministic=args.deterministic,
        debug=args.debug,
        precision=args.precision,
    )

    generator = Generator(generator, LATENT_DIM)

    victim_model = VictimModel(victim_model, NUM_CLASSES, output_type="softmax")
    substitute_model = TrainableModel(
        substitute_model,
        NUM_CLASSES,
        torch.optim.Adam,
        soft_cross_entropy,
        batch_accuracy=True,
    )

    return victim_model, substitute_model, generator, test_set


if __name__ == "__main__":
    parser = Ripper.get_attack_args()
    parser.add_argument(
        "generator_checkpoint",
        type=str,
        help="Location of torch_mimicry checkpoint " "for SNGAN.",
    )
    parser.add_argument(
        "--fashion_mnist_dir",
        default="./data/",
        type=str,
        help="Path to CIFAR10 dataset (Default: ./data/)",
    )
    args = parser.parse_args()
    args.training_epochs = 200
    args.batch_size = 64

    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, generator, test_set = set_up(args)
    rp = Ripper(
        victim_model,
        substitute_model,
        generator,
        args.generated_data,
        args.batches_per_epoch,
        args.population_size,
        args.max_iterations,
        args.threshold_type,
        args.threshold_value,
        args.find_additional,
        args.save_dataset,
        args.dataset_save_loc,
    )

    # Baset settings
    rp.base_settings.save_loc = Path(args.save_loc)
    rp.base_settings.gpu = args.gpu
    rp.base_settings.num_workers = args.num_workers
    rp.base_settings.batch_size = args.batch_size
    rp.base_settings.seed = args.seed
    rp.base_settings.deterministic = args.deterministic
    rp.base_settings.debug = args.debug

    # Trainer settings
    rp.trainer_settings.training_epochs = args.training_epochs
    rp.trainer_settings.evaluation_frequency = 1
    rp.trainer_settings.precision = args.precision
    rp.trainer_settings.use_accuracy = args.accuracy

    rp(test_set, test_set)
