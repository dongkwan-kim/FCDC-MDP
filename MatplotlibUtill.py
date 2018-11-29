import matplotlib.pyplot as plt
import numpy as np


def build_hist(list_of_enumerable, labels, title, xlabel=None, ylabel=None, **kwargs):

    fig, ax = plt.subplots(figsize=(5, 4))

    for enumerable, label in zip(list_of_enumerable, labels):
        n, bins, patches = ax.hist(
            list(enumerable),
            label=label,
            **kwargs,
        )

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

    return plt


if __name__ == '__main__':

    xs = [
        np.random.normal(5, 1, size=100),
        np.random.normal(3, 2, size=100)
    ]

    build_hist(xs, ["a", "b"],
               title="wow", xlabel="x", ylabel="y",
               bins=3, cumulative=True, histtype="step")
