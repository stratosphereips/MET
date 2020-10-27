import argparse


def get_default_parser(description):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-v", "--victim_train_epochs", default=10, type=int,
                        help="Number of epochs for which the victim should "
                             "train for (Default: 10)")
    parser.add_argument("-r", "--seed", default=0, type=int,
                        help="Random seed to be used (Default: 0)")
    parser.add_argument("-t", "--training_epochs", default=1000, type=int,
                        help="Number of training epochs for substitute model "
                             "(Default: 1000)")
    parser.add_argument("-e", "--early_stop_tolerance", default=10, type=int,
                        help="Number of epochs without improvement for early "
                             "stop (Default: 10)")
    parser.add_argument("-a", "--evaluation_frequency", default=2, type=int,
                        help="Epochs interval of validation (Default: 2)")
    parser.add_argument("-s", "--val_size", type=float, default=0.2,
                        help="Size of validation set (Default: 0.2)")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size to be used (Default: 32)")
    parser.add_argument("-l", "--save_loc", type=str, default="./cache/",
                        help="Path where the attacks file should be "
                             "saved (Default: ./cache/)")
    parser.add_argument("-g", "--gpus", type=int, default=0,
                        help="Number of gpus to be used (Default: 0)")
    parser.add_argument("-d", "--deterministic", action="store_true",
                        help="Run in deterministic mode (Default: True)")
    parser.add_argument("-u", "--debug", action="store_true",
                        help="Run in debug mode (Default: False)")

    return parser
