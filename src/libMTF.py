import os
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
import imageio

TIFF_DEFLATE = 32946


def create_mtf(
    segment,
    n_bins=8,  # Number of bins for discretization
    images_dir="",
    base_name="sample",
    suffix="tif",  # suffix='jpg', # suffix='png'
    compress=TIFF_DEFLATE,
    show_image=False,
    cmap=None,  # cmap='gray', cmap='binary'
):
    """Generate Markov Transition Field for specified signal segment and save to disk"""

    if base_name is None:
        base_name = "sample"
    fname = "{}_mtf_{}.{}".format(base_name, n_bins, suffix)

    segment = np.expand_dims(segment, 0)

    mtf = MarkovTransitionField(n_bins=n_bins)
    X_mtf = mtf.fit_transform(segment)[0]

    imageio.imwrite(
        os.path.join(images_dir, fname),
        np_to_uint8(X_mtf),
        format=suffix,
        **{"compression": compress}
    )

    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(X_mtf, cmap=cmap, origin="lower")
        plt.title("Markov Transition Field for {}".format(fname), fontsize=14)
        plt.gca().invert_yaxis()
        plt.show()
    return fname


def np_to_uint8(X):
    X -= X.min()
    X = (255 / X.max()) * X
    return X.astype(np.uint8)
