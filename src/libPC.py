#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter

TIFF_DEFLATE = 32946


def calculate_rr(frh):

    rr = 60*1000 / frh

    return rr


def plot_poincare(data):

    rr = calculate_rr(data)

    plt.scatter(rr[:-1], rr[1:], s=4, marker='s')

    plt.axis('off')
    plt.axis('tight')  # gets rid of white border
    plt.axis('image')

    fig = plt.gcf()
    fig.canvas.draw()
    array_data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    return array_data


def create_pc(segment,
              use_clip=False, knn=None, imsize=None,
              images_dir='', base_name='Sample',
              suffix='tif',  # suffix='jpg', # suffix='png'
              compress=TIFF_DEFLATE,
              show_image=False, cmap=None,  # cmap='gray', cmap='binary'
              ):
    """Generate poincaré plot for specified signal segment and save to disk"""

    if base_name is None:
        base_name = 'sample'
    fname = '{}_pc{}.{}'.format(
        base_name, '_clipped' if use_clip else '', suffix)

    # segment = np.expand_dims(segment, 0)
    # pc = plot_poincare(segment)
    pc = msp_plots(segment)

    imageio.imwrite(os.path.join(images_dir, fname), pc,
                    format=suffix, **{"compression": compress})

    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(pc, cmap=cmap, origin='lower')
        plt.title('Poincaré Plot for {}'.format(fname), fontsize=14)
        plt.xlabel('RR$_n$ (ms)')
        plt.ylabel('RR$_{n+1}$ (ms)')
        plt.show()

    return fname


def np_to_uint8(X):
    X -= X.min()
    X = (255/X.max())*X
    return X.astype(np.uint8)


def msp_plots(X, scales=1, scstep=1, nrow=1, ncol=1, save=False):
    # def msp_plots(X, scales=12, scstep=1, nrow=3, ncol=4, save=False):
    """
    MSP function creates an ensemble of Poincare Plots, one for each coarse grained time series

    Parameters:
        X (np.ndarray): a time series vector with one column.
        scales (int, optional): the number of time scales. Default is 12.
        scstep (int, optional): Poincare plots from 1 to scale by scstep. Default is 1.
        nrow (int, optional): number of rows in the plot montage. Default is 3.
        ncol (int, optional): number of columns in the plot montage. Default is 4.
        save (bool, optional): if True, saves the figure as 'Figure1.jpg'. Default is True.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 1:
        raise ValueError('X must be a numeric vector')

    if not isinstance(scales, int) or scales <= 0:
        raise ValueError('scales must be a positive integer')

    if not isinstance(scstep, int) or scstep <= 0:
        raise ValueError('scstep must be a positive integer')

    if not isinstance(nrow, int) or nrow <= 0:
        raise ValueError('nrow must be a positive integer')

    if not isinstance(ncol, int) or ncol <= 0:
        raise ValueError('ncol must be a positive integer')

    S0 = range(1, scales + 1, scstep)

    if len(S0) > nrow * ncol:
        raise ValueError(
            'Adjust the number of plots by number of rows and columns to be displayed')

    Xmin = np.floor(np.min(X) * 10) / 10 - 0.05
    Xmax = np.ceil(np.max(X) * 10) / 10 + 0.05
    Xstep = round((Xmax - Xmin) / 5 * 10) / 10
    ticks = np.arange(Xmin, Xmax + Xstep, Xstep)

    fig, axes = plt.subplots(nrow, ncol, figsize=(7, 7))
    if nrow * ncol == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, s in enumerate(S0):
        ys = coarse_grain(X, s)
        yp = ys[:-1]
        ym = ys[1:]

        ax = axes[idx]
        # sc = ax.scatter(yp, ym, c='b', alpha=0.5, edgecolors='none')
        sc = dscatter(yp, ym, smoothing=20, bins=[
            700, 700], plottype='scatter', marker='s', msize=3, filled=True)

        # ax.set_xlim([Xmin, Xmax])
        # ax.set_ylim([Xmin, Xmax])
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        # ax.grid(True)
        # ax.set_title(f'z_{s}(i) vs z_{s}(i+1)')
        # ax.set_xlabel(f'z_{s}(i)')
        # ax.set_ylabel(f'z_{s}(i+1)')

    for ax in axes[len(S0):]:
        fig.delaxes(ax)

    if save:
        plt.savefig('../content/images/Figure1.jpg', dpi=300,
                    pad_inches=0, bbox_inches='tight')

    fig = plt.gcf()
    fig.canvas.draw()
    array_data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    return array_data


def coarse_grain(x, s):
    """
    Creates Coarse Grain time series for scale s by averaging s consecutive non-overlapping data points

    Parameters:
        x (np.ndarray): input time series
        s (int): scale factor

    Returns:
        np.ndarray: coarse-grained time series
    """
    L = len(x)
    Jmax = L // s
    y_s = np.zeros(Jmax)

    for j in range(Jmax):
        ind1 = j * s
        ind2 = (j + 1) * s
        y_s[j] = np.mean(x[ind1:ind2])

    return y_s


# Example usage:
# X = np.random.randn(1000)
# msp_plots(X)


# df = pd.read_csv('../content/1005_60min.csv')

# length = 10 * 4 * 60
# # rr_intervals = df['hr'].tail(length).head(2400).values
# rr_intervals = df['hr'].tail(length).values

# msp_plots(rr_intervals, 1, 1, 1, 1)


def dscatter(X, Y, **kwargs):
    # Default parameters
    lambda_val = 20
    nbins = None
    plottype = 'scatter'
    logy = False
    msize = 10
    marker = 's'
    filled = True

    # Handle additional arguments
    for key, value in kwargs.items():
        if key == 'smoothing':
            lambda_val = value
        elif key == 'bins':
            nbins = value if isinstance(value, (list, tuple)) else [
                value, value]
        elif key == 'plottype':
            plottype = value
        elif key == 'logy':
            logy = value
            if logy:
                Y = np.log10(Y)
        elif key == 'marker':
            marker = value
        elif key == 'msize':
            msize = value
        elif key == 'filled':
            filled = value
        else:
            raise ValueError(f"Unknown parameter: {key}")

    # Calculate the bin edges and centers
    if nbins is None:
        nbins = [min(len(np.unique(X)), 200), min(len(np.unique(Y)), 200)]

    minx, maxx = np.min(X), np.max(X)
    miny, maxy = np.min(Y), np.max(Y)
    edges1 = np.linspace(minx, maxx, nbins[0] + 1)
    ctrs1 = edges1[:-1] + 0.5 * np.diff(edges1)
    edges2 = np.linspace(miny, maxy, nbins[1] + 1)
    ctrs2 = edges2[:-1] + 0.5 * np.diff(edges2)

    # Digitize the data
    bin1 = np.digitize(X, edges1) - 1
    bin2 = np.digitize(Y, edges2) - 1

    # Ensure bins are within the range
    bin1[bin1 >= nbins[0]] = nbins[0] - 1
    bin2[bin2 >= nbins[1]] = nbins[1] - 1

    H, _, _ = np.histogram2d(Y, X, bins=nbins)

    # Smooth the histogram
    H = H / H.sum()
    H = gaussian_filter(H, sigma=lambda_val)

    if logy:
        ctrs2 = 10 ** ctrs2

    plt.axis('off')

    if plottype == 'surf':
        Xgrid, Ygrid = np.meshgrid(ctrs1, ctrs2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xgrid, Ygrid, H, edgecolor='none')
    elif plottype == 'mesh':
        Xgrid, Ygrid = np.meshgrid(ctrs1, ctrs2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(Xgrid, Ygrid, H)
    elif plottype == 'contour':
        plt.contour(ctrs1, ctrs2, H)
    elif plottype == 'image':
        plt.imshow(H, extent=[minx, maxx, miny, maxy],
                   origin='lower', aspect='auto')
        plt.colorbar()
    elif plottype == 'scatter':
        F = H.flatten()
        ind = bin2 * nbins[0] + bin1
        col = F[ind]

        plt.scatter(X, Y, c=col, s=msize, marker=marker, cmap='jet',
                    edgecolor='none' if filled else 'k')

    if logy:
        plt.yscale('log')

    plt.tight_layout()
    plt.margins(0, 0)
