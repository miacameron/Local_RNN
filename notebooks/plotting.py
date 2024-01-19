import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
import matplotlib.animation as animation
import math
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_compare(rec, ori, title=" ", rec_title=" ", ori_title=" ", loss=0):
    plt.figure(figsize=[20, 5])
    plt.suptitle(title, fontsize=16)

    plt.subplot(131)
    plt.pcolormesh(rec.T, cmap="viridis")
    plt.title(rec_title)
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(132)
    plt.pcolormesh(ori.T, cmap="viridis")
    plt.title(ori_title)
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(133)
    plt.pcolormesh((rec - ori).T, cmap="seismic", norm=CenteredNorm())
    plt.title("MSE: " + str(round(loss, 4)))
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.colorbar()
    plt.show()


def plot_input(x, name=""):
    plt.pcolormesh(x.T, cmap="viridis")
    plt.title(name)
    plt.xlabel("time")
    plt.ylabel("neuron #")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()


def plot_weights(model):
    plt.figure(figsize=[20, 5])

    plt.subplot(131)
    plt.pcolormesh(model.W_f.T, cmap="bwr", norm=CenteredNorm())
    plt.title("$W_f$")
    plt.colorbar()

    plt.subplot(132)
    plt.pcolormesh(model.W_r.T, cmap="bwr", norm=CenteredNorm())
    plt.title("$W_r$")
    plt.colorbar()

    plt.subplot(133)
    plt.pcolormesh(model.W_g.T, cmap="bwr", norm=CenteredNorm())
    plt.title("$W_g$")
    plt.colorbar()
    plt.show()


def plot_digit(x):
    N = x.shape[0]
    n = int(math.sqrt(N))
    plt.imshow(x.reshape(n, n), cmap="binary_r")
    plt.colorbar()
    plt.show()


def plot_digits(X):
    N = X.shape[0]
    fig,axes = plt.subplots(1,N,figsize=[N*2, 2])
    c_min = np.min(X)
    c_max = np.max(X)
    for ax,n in zip(axes,range(0,N)):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        ax.imshow(X[n].reshape(28, 28), cmap="binary_r",vmin=c_min,vmax=c_max)
    im = ax.imshow(X[0].reshape(28, 28), cmap="binary_r",vmin=c_min,vmax=c_max)
    fig.colorbar(im,ax=axes.ravel().tolist())
    plt.show(block=True)

def plot_digits_grid(X):
    N = X.shape[0]
    fig = plt.figure(figsize=(N*2,2))
    c_min = 0
    c_max = 1
    grid = ImageGrid(fig, (0,0,N,1),
                    nrows_ncols=(1,N),
                    axes_pad=0.15,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )
    for ax,n in zip(grid,range(0,N)):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        im = ax.imshow(X[n].reshape(28, 28), cmap="binary_r",vmin=c_min,vmax=c_max)
    ax.cax.colorbar(im)
    ax.cax.tick_params(labelsize=20)
    ax.cax.toggle_label(True)
    plt.show()


def plot_compare_digits(X_ori, X_rec):
    N = X_ori.shape[0]
    fig = plt.figure(figsize=[N * 3 + 3, 9])
    row = fig.subfigures(nrows=3, ncols=1)
    # fig.tight_layout()

    # Plot original X
    row[0].suptitle("Original", fontsize=16)
    col0 = row[0].subplots(nrows=1, ncols=N, sharey=True)
    c0 = col0[0].imshow(X_ori[0].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
    row[0].colorbar(c0, ax=col0, shrink=0.8)
    for n in range(1, N):
        col0[n].imshow(X_ori[n].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
        # col0[n].tick_params(left=False, bottom=False)

    # Plot recalled X
    row[1].suptitle("Recalled", fontsize=16)
    col1 = row[1].subplots(nrows=1, ncols=N, sharey=True)
    c1 = col1[0].imshow(X_rec[0].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
    row[1].colorbar(c1, ax=col1, shrink=0.8)
    for n in range(1, N):
        col1[n].imshow(X_rec[n].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
        # col1[n].tick_params(left=False, bottom=False)

    # Plot X error
    row[2].suptitle("Difference", fontsize=16)
    col2 = row[2].subplots(nrows=1, ncols=N, sharey=True)
    c2 = col2[0].imshow(
        (X_ori[0] - X_rec[0]).reshape(28, 28), cmap="bwr", vmin=-0.1, vmax=0.1
    )
    row[2].colorbar(c2, ax=col2, shrink=0.8)
    for n in range(1, N):
        col2[n].imshow(
            (X_ori[n] - X_rec[n]).reshape(28, 28), cmap="bwr", vmin=-0.1, vmax=0.1
        )
        # col2[n].tick_params(left=False, bottom=False)

    # plt.tight_layout()
    plt.show()


def plot_moving_digits(X_ori, X_rec):
    N = X_ori.shape[0]
    fig = plt.figure(figsize=[N * 2, 6])
    row = fig.subfigures(nrows=2, ncols=1)
    # fig.tight_layout()
    c_max = np.max(X_ori) * 2
    D = 64

    # Plot original X
    row[0].suptitle("Original", fontsize=16)
    col0 = row[0].subplots(nrows=1, ncols=N, sharey=True)
    c0 = col0[0].imshow(X_ori[0].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)
    row[0].colorbar(c0, ax=col0, shrink=0.8)
    for n in range(1, N):
        col0[n].imshow(X_ori[n].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)

    # Plot recalled X
    row[1].suptitle("Recalled", fontsize=16)
    col1 = row[1].subplots(nrows=1, ncols=N, sharey=True)
    c1 = col1[0].imshow(X_rec[0].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)
    row[1].colorbar(c1, ax=col1, shrink=0.8)
    for n in range(1, N):
        col1[n].imshow(X_rec[n].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)

    plt.show()


def animate_imgs(X, title="X", c_max=1, diff=False):
    fig, ax = plt.subplots()
    T = X.shape[0]
    D = 64
    c_min = 0
    # c_max = np.max(X)*2
    c_map = "viridis"
    if diff:
        c_min = -c_max
        c_map = "bwr"

    imgs = []
    for t in range(T):
        im = ax.imshow(X[t, :].reshape(D, D), cmap=c_map, vmin=c_min, vmax=c_max)
        imgs.append([im])

    mov = animation.ArtistAnimation(
        fig, imgs, interval=250, blit=True, repeat_delay=1000
    )
    mov.save(title + ".mp4")


def animate_3_imgs(X, X_rec, title="X"):
    fig, ax = plt.subplots()
    p0 = fig.add_subplot(131)
    p1 = fig.add_subplot(132)
    p2 = fig.add_subplot(133)

    T = X.shape[0]
    D = 64
    c_min = 0
    c_max = np.max(X) * 2

    X_diff = X - X_rec

    imgs = []
    for t in range(T):
        im0 = p0.imshow(X[t, :].reshape(D, D), cmap="viridis", vmin=c_min, vmax=c_max)

        im1 = p1.imshow(
            X_rec[t, :].reshape(D, D), cmap="viridis", vmin=c_min, vmax=c_max
        )

        im2 = p2.imshow(X_diff[t, :].reshape(D, D), cmap="bwr", vmin=-c_max, vmax=c_max)
        imgs.append(ax.get_images())
        ax.clear()

    m = animation.ArtistAnimation(
        fig, imgs, interval=250, blit=False, repeat_delay=1000
    )

    m.save(title + ".mp4")
    plt.show(m)


def plot_lines(X):
    N = X.shape[0]

    plt.figure(figsize=[N * 3, 3])

    for n in range(N):
        idx = 100 + 10 * N + (n + 1)
        plt.subplot(idx)
        plt.imshow(X[n], cmap="binary_r")

    plt.show()
