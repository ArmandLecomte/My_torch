import argparse
import os.path
from pathlib import Path
from sys import stderr

from my_torch import NNGenerator

class EpitechParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(stderr)
        self.exit(84, '%s: error: %s\n' % (self.prog, message))

def generate_models(configs: dict[str, int]):
    for config_file, count in configs.items():
        for j in range(count):
            model = NNGenerator.generate_neural_network(config_file)
            config_name = config_file.rsplit(".", 1)[0]
            model.save(f"{config_name}_{j + 1}.nn")


def main():
    parser = EpitechParser(
        prog="./my_torch_generator",
        usage="%(prog)s config_file_1 nb_1 [config_file_2 nb_2...]",
    )
    parser.add_argument("config_file", metavar="config_file_i",
                        help="Configuration file containing description of a neural network we want to generate.")
    parser.add_argument("nb", metavar="nb_i",
                        help="Number of neural networks to generate based on the configuration file.")
    parser.add_argument("rest", nargs="*",
                        help=argparse.SUPPRESS)  # paires suivantes, masquées de l'aide
    args = parser.parse_args()

    values = [args.config_file, args.nb] + args.rest
    if len(values) % 2 != 0:
        parser.error("each config_file must be followed by a number")

    configs = {}

    for i in range(0, len(values), 2):
        config_file = values[i]
        count = values[i + 1]
        try:
            count = int(count)
        except ValueError:
            parser.error(f"Invalid number of neural networks: {count}")
        if count <= 0:
            parser.error(f"Number of neural networks must be positive: {count}")
        if not os.path.isfile(config_file):
            parser.error(f"Configuration file does not exist: {config_file}")
        configs[config_file] = count

    generate_models(configs)


if __name__ == "__main__":
    main()
