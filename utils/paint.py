import numpy as np
from matplotlib import pyplot as plt


def paint_amap(acmap, path=None, dpi=100, save=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(512. / dpi, 512. / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(np.asanyarray(acmap.squeeze(), dtype=np.float32), vmin=-0.5, vmax=0.5, cmap='Spectral_r')
    # plt.colorbar(orientation='horizontal', ticks=[-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    if save:
        plt.savefig(path, dpi=dpi)

def paint(x, cmap, path=None, dpi=100, save=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(512./dpi, 512./dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(np.asanyarray(x.squeeze(), dtype=np.float32), cmap=cmap)
    if save:
        plt.savefig(path, dpi=dpi)