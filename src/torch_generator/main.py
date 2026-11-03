import argparse

from my_torch import ConstantLR

if __name__ == "__main__":

    lr = 10
    lr_scheduler = ConstantLR(lr)
    data = []
    for i in range(1000):
        lr = lr_scheduler.get_learning_rate(lr, i)
        data.append((i, lr))

    import matplotlib.pyplot as plt
    plt.plot(*zip(*data))
    plt.show()
    parser = argparse.ArgumentParser(
        prog="./my_torch_generator",
        usage="%(prog)s config_file_1 nb_1 [config_file_2 nb_2...]",
    )
    parser.add_argument("config_file", metavar="config_file_i",
                        help="Configuration file containing description of a neural network we want to generate.")
    parser.add_argument("nb", metavar="nb_i",
                        help="Number of neural networks to generate based on the configuration file.")
    parser.add_argument("rest", nargs="*", help=argparse.SUPPRESS)  # paires suivantes, masquées de l'aide
    args = parser.parse_args()

    values = [args.config_file, args.nb] + args.rest
    if len(values) % 2 != 0:
        parser.error("each config_file must be followed by a number")


