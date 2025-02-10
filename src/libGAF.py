import os
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import imageio

# from PIL import Image, TiffImagePlugin


TIFF_DEFLATE = 32946


def create_gaf(
    segment,
    method="summation",  # 'summation' or 'difference'
    images_dir="",
    base_name="sample",
    suffix="tif",  # suffix='jpg', # suffix='png'
    compress=TIFF_DEFLATE,
    show_image=False,
    cmap=None,  # cmap='gray', cmap='binary'
):
    """Generate Gramian Angular Field for specified signal segment and save to disk"""

    if base_name is None:
        base_name = "sample"
    fname = "{}_gaf_{}.{}".format(base_name, method, suffix)

    segment = np.expand_dims(segment, 0)

    gaf = GramianAngularField(method=method)
    X_gaf = gaf.fit_transform(segment)[0]

    imageio.imwrite(
        os.path.join(images_dir, fname),
        np_to_uint8(X_gaf),
        format=suffix,
        **{"compression": compress}
    )

    # img = Image.fromarray(np_to_uint8(X_gaf))
    # img.save(os.path.join(images_dir, fname), format="TIFF", compression="tiff_deflate")

    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(X_gaf, cmap=cmap, origin="lower")
        plt.title("Gramian Angular Field for {}".format(fname), fontsize=14)
        plt.gca().invert_yaxis()
        plt.show()
    return fname


def np_to_uint8(X):
    X -= X.min()
    X = (255 / X.max()) * X
    return X.astype(np.uint8)
