import argparse
import pickle
import sys
from argparse import ArgumentParser, Namespace

import numpy as np

from my_torch import NeuralNetwork
from my_torch.typing import DataPair
from parsing.fen_parser import FENParseError, FENParser, FENPosition


_PIECE_ORDER = "PNBRQKpnbrqk"
_CLASS_NAMES = {
    3: ("Nothing", "Check", "Checkmate"),
    5: ("Nothing", "Check White", "Check Black",
        "Checkmate White", "Checkmate Black"),
    7: ("Nothing", "Check", "Checkmate", "Check White", "Check Black",
        "Checkmate White", "Checkmate Black"),
}
_EPOCHS = 100
_LEARNING_RATE = 0.05


class _ArgumentParser(ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(84, f"{self.prog}: error: {message}\n")


def encode_position(position: FENPosition) -> np.ndarray:
    encoded = np.zeros(64 * len(_PIECE_ORDER) + 2, dtype=np.float64)
    square = 0
    for rank in position.board:
        for piece in rank:
            if piece is not None:
                feature = square * len(_PIECE_ORDER) + _PIECE_ORDER.index(piece)
                encoded[feature] = 1.0
            square += 1
    if square != 64:
        raise ValueError("Parsed position does not contain 64 squares")
    active_color_index = 64 * len(_PIECE_ORDER) + (
        0 if position.active_color == "w" else 1
    )
    encoded[active_color_index] = 1.0
    return encoded


def _load_network(path: str) -> NeuralNetwork:
    try:
        network = NeuralNetwork.load(path)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError,
            ImportError, ValueError) as error:
        raise ValueError(f"cannot load neural network {path!r}: {error}") from error
    if not isinstance(network, NeuralNetwork):
        raise ValueError(f"{path!r} does not contain a neural network")
    return network


def _class_names_for(network: NeuralNetwork) -> tuple[str, ...]:
    output = network.forward(np.zeros((1, 64 * len(_PIECE_ORDER) + 2)))
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("neural network must return one row of class scores")
    class_names = _CLASS_NAMES.get(output.shape[1])
    if class_names is None:
        raise ValueError(
            f"unsupported neural network output size: {output.shape[1]} "
            "(expected 3, 5, or 7)"
        )
    return class_names


def execute_train(args: Namespace) -> int:
    parser = FENParser()
    records = parser.parse_file(args.CHESSFILE)
    if not records:
        raise ValueError("training file does not contain any chessboards")

    network = _load_network(args.LOADFILE)
    class_names = _class_names_for(network)
    training_data: list[DataPair] = []
    for record in records:
        label = record.expected_result
        if label is None:
            raise ValueError("training positions must include an expected result")
        try:
            class_index = class_names.index(label)
        except ValueError as error:
            raise ValueError(
                f"expected result {label!r} is not supported by this network"
            ) from error

        inputs = encode_position(record.position).reshape(1, -1)
        targets = np.zeros((1, len(class_names)), dtype=np.float64)
        targets[0, class_index] = 1.0
        training_data.append(DataPair(inputs, targets))

    network.train(
        training_data,
        num_epoch=_EPOCHS,
        learning_rate=_LEARNING_RATE,
    )
    network.save(args.save or args.LOADFILE)
    return 0


def execute_predict(args: Namespace) -> int:
    parser = FENParser()
    records = parser.parse_file(args.CHESSFILE)
    network = _load_network(args.LOADFILE)
    class_names = _class_names_for(network)

    for record in records:
        scores = network.forward(encode_position(record.position).reshape(1, -1))
        print(class_names[int(np.argmax(scores[0]))])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _ArgumentParser(
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
    args = parser.parse_args(argv)
    if args.save and not args.train:
        parser.error("--save can only be used with --train")

    try:
        if args.train:
            return execute_train(args)
        return execute_predict(args)
    except (FENParseError, OSError, ValueError) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 84


if __name__ == "__main__":
    sys.exit(main())
