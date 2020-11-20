import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torchvision.transforms import transforms as T

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from mef.attacks.knockoff import KnockOff
from mef.utils.experiment import train_victim_model
from mef.utils.ios import mkdir_if_missing
from mef.utils.pytorch.datasets.vision import ImageNet1000, Caltech256
from mef.utils.pytorch.models.vision import ResNet

NUM_CLASSES = 256


def set_up(args):
    seed_everything(args.seed)

    victim_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)
    substitute_model = ResNet(resnet_type="resnet_34", num_classes=NUM_CLASSES)

    if args.gpus:
        victim_model.cuda()
        substitute_model.cuda()

    # Prepare data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize(mean, std)])

    train_set = Caltech256(args.caltech256_dir, transform=transform,
                           seed=args.seed)
    test_set = Caltech256(args.caltech256_dir, train=False,
                          transform=transform, seed=args.seed)
    sub_dataset = ImageNet1000(args.imagenet_dir, transform=transform,
                               seed=args.seed)

    optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.1,
                                momentum=0.5)
    loss = F.cross_entropy
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60)

    train_victim_model(victim_model, optimizer, loss, train_set,
                       NUM_CLASSES, args.precision, args.batch_size,
                       lr_scheduler=lr_scheduler, save_loc=args.save_loc,
                       gpus=args.gpus, deterministic=args.deterministic,
                       debug=args.debug, precision=args.precision)

    return victim_model, substitute_model, sub_dataset, test_set


if __name__ == "__main__":
    parser = KnockOff.get_attack_args()
    parser.add_argument("--caltech256_dir", default="./data/", type=str,
                        help="Path to Caltech256 dataset (Default: ./data/")
    parser.add_argument("--imagenet_dir", type=str,
                        help="Path to ImageNet dataset")
    parser.add_argument("--victim_train_epochs", default=200, type=int,
                        help="Number of epochs for which the victim should "
                             "train for (Default: 200)")
    args = parser.parse_args()
    mkdir_if_missing(args.save_loc)

    victim_model, substitute_model, sub_dataset, test_set = set_up(args)
    ko = KnockOff(victim_model, substitute_model, NUM_CLASSES,
                  args.sampling_strategy, args.reward_type,
                  args.victim_output_type, args.budget)

    # Baset settings
    ko.base_settings.save_loc = Path(args.save_loc)
    ko.base_settings.gpus = args.gpus
    ko.base_settings.seed = args.seed
    ko.base_settings.deterministic = args.deterministic
    ko.base_settings.debug = args.debug

    # Trainer settings
    ko.trainer_settings.training_epochs = args.training_epochs
    ko.trainer_settings.precision = args.precision
    ko.trainer_settings.accuracy = args.accuracy

    # Data settings
    ko.data_settings.batch_size = args.batch_size

    ko(sub_dataset, test_set)
