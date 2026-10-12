import argparse
import sys
from argparse import ArgumentParser, Namespace


def execute_train(args: Namespace) -> int:
    return 0


def execute_predict(args: Namespace) -> int:
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="./my_torch_analyzer",
        usage="./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE",
    )

    desc = parser.add_argument_group("DESCRIPTION")

    mode = desc.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true",
                      help="Launch the neural network in training mode. Each chessboard in FILE must "
                           "contain inputs to send to the neural network in FEN notation and the expected "
                           "output separated by space. If specified, the newly trained neural network will "
                           "be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.")
    mode.add_argument("--predict", action="store_true",
                      help="Launch the neural network in prediction mode. Each chessboard in FILE must "
                           "contain inputs to send to the neural network in FEN notation, and optionally "
                           "an expected output.")

    desc.add_argument("--save", metavar="SAVEFILE",
                      help="Save neural network into SAVEFILE. Only works in train mode.")
    desc.add_argument("LOADFILE", help="File containing an artificial neural network")
    desc.add_argument("CHESSFILE", help="File containing chessboards")
    args = parser.parse_args()

    if args.train:
        sys.exit(execute_train(args))
    sys.exit(execute_predict(args))