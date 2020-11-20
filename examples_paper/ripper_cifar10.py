import os
import sys
from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.utils.pytorch.models.generators.cifar_sngan.sngan_cifar10 import \
    Generator as SNGAN
from mef.attacks.ripper import Ripper
from mef.utils.pytorch.models.vision import Alexnet, HalfAlexnet
from mef.utils.ios import mkdir_if_missing

IMAGENET_TRAIN_SIZE = 100000
LATENT_DIM = 128
DIMS = (3, 32, 32)
NUM_CLASSES = 10


def set_up(args):
    seed_everything(args.seed)

    victim_model = Alexnet("teacher_alexnet_for_cifar10", 10)
    substitute_model = HalfAlexnet(
            "student_half_alexnet_for_teacher_alexnet_true_cifar10_generator_cifar_100_90_classes_gan_samples_optimized_optim_adam_epochs_200",
            10)
    generator = SNGAN()

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()
        generator.cuda()

    # Prepare data
    print("Preparing data")
    mean = (0.5,)
    std = (0.5,)
    transform = T.Compose([T.Resize(DIMS[-1]), T.ToTensor(),
                           T.Normalize(mean, std)])
    test_set = CIFAR10(root=args.cifar10_dir, train=False, download=True,
                       transform=transform)

    # Load state dicts
    state_dict = torch.load("./cache/ripper/CIFAR10/victim/"
                            "teacher_alexnet_for_cifar10_state_dict")
    victim_model.load_state_dict(state_dict)
    state_dict = torch.load("./cache/ripper/CIFAR10/generator/"
                            "cifar_100_90_classes_gan.pth")["gen_state_dict"]
    generator.load_state_dict(state_dict)

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
    rp = Ripper(victim_model, substitute_model, generator, LATENT_DIM,
                NUM_CLASSES, args.generated_data, args.output_type)

    # Baset settings
    rp.base_settings.save_loc = Path(args.save_loc)
    rp.base_settings.gpus = args.gpus
    rp.base_settings.seed = args.seed
    rp.base_settings.deterministic = args.deterministic
    rp.base_settings.debug = args.debug

    # Trainer settings
    rp.trainer_settings.training_epochs = args.training_epochs
    rp.trainer_settings.precision = args.precision
    rp.trainer_settings.accuracy = args.accuracy

    # Data settings
    rp.data_settings.batch_size = args.batch_size

    rp(test_set)
