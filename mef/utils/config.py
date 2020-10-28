import argparse


def add_activethief_arguments(parser):
    parser.add_argument("--selection_strategy", default="entropy",
                        type=str, help="Activethief selection strategy can "
                                       "be one of {random, entropy, k-center, "
                                       "dfal, dfal+k-center} (Default: "
                                       "entropy)")
    parser.add_argument("--iterations", default=10, type=int,
                        help="Number of iterations of the attacks (Default: "
                             "10)")
    parser.add_argument("--output_type", default="softmax", type=str,
                        help="Type of output from victim model {softmax, "
                             "logits, one_hot} (Default: softmax)")
    parser.add_argument("--init_seed_size", default=2000, type=int,
                        help="Size of the initial random query set (Default: "
                             "2000)")
    parser.add_argument("--budget", default=20000, type=int,
                        help="Size of the budget (Default: 20000)")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed to be used (Default: 0)")
    parser.add_argument("--substitute_train_epochs", default=1000, type=int,
                        help="Number of training epochs for substitute model "
                             "(Default: 100)")
    parser.add_argument("--early_stop_tolerance", default=100, type=int,
                        help="Number of epochs without improvement for early "
                             "stop (Default: 10)")
    parser.add_argument("--evaluation_frequency", default=2, type=int,
                        help="Epochs interval of validation (Default: 2)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Size of validation set (Default: 0.2)")

    return


def add_knockoff_arguments(parser):
    parser.add_argument("--sampling_strategy", default="adaptive",
                        type=str, help="KnockOff-Nets sampling strategy can "
                                       "be one of {random, adaptive} ("
                                       "Default: adaptive)")
    parser.add_argument("--reward_type", default="all", type=str,
                        help="Type of reward for adaptive strategy, can be "
                             "one of {cert, div, loss, all} (Default: all)")
    parser.add_argument("--output_type", default="softmax", type=str,
                        help="Type of output from victim model {softmax, "
                             "logits, one_hot} (Default: softmax)")
    parser.add_argument("--budget", default=20000, type=int,
                        help="Size of the budget (Default: 20000)")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed to be used (Default: 0)")
    parser.add_argument("--substitute_train_epochs", default=100, type=int,
                        help="Number of training epochs for substitute model "
                             "(Default: 100)")
    parser.add_argument("--early_stop_tolerance", default=10, type=int,
                        help="Number of epochs without improvement for early "
                             "stop (Default: 10)")

    return


def add_copycat_arguments(parser):
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed to be used (Default: 0)")
    parser.add_argument("--substitute_train_epochs", default=1000, type=int,
                        help="Number of training epochs for substitute model "
                             "(Default: 100)")
    parser.add_argument("--early_stop_tolerance", default=100, type=int,
                        help="Number of epochs without improvement for early "
                             "stop (Default: 10)")
    parser.add_argument("--evaluation_frequency", default=2, type=int,
                        help="Epochs interval of validation (Default: 2)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Size of validation set (Default: 0.2)")

    return

def add_blackbox_arguments(parser):
    parser.add_argument("--iterations", default=6, type=int,
                        help="Number of iterations of the attacks (Default: "
                             "6)")
    parser.add_argument("--lmbda", default=0.1, type=float,
                        help="Value of lambda in Jacobian augmentation ("
                             "Default: 0.1)")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed to be used (Default: 0)")
    parser.add_argument("--substitute_train_epochs", default=10, type=int,
                        help="Number of training epochs for substitute model "
                             "(Default: 100)")

    return


def get_attack_parser(description, attack_type):
    parser = argparse.ArgumentParser(description=description)

    attack_type = attack_type.lower()
    if attack_type == "activethief":
        add_activethief_arguments(parser)
    elif attack_type == "knockoff":
        add_knockoff_arguments(parser)
    elif attack_type == "copycat":
        add_copycat_arguments(parser)
    elif attack_type == "blackbox":
        add_blackbox_arguments(parser)
    else:
        raise ValueError("type must be one of {activethief, knockoff, "
                         "copycat, blackbox}")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size to be used (Default: 64)")
    parser.add_argument("--save_loc", type=str, default="./cache/",
                        help="Path where the attacks file should be "
                             "saved (Default: ./cache/)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of gpus to be used (Default: 0)")
    parser.add_argument("--deterministic", action="store_false",
                        help="Run in deterministic mode (Default: True)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (Default: False)")

    return parser
