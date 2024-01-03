
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import seaborn as sns

TIFF_DEFLATE = 32946

plt.ioff()


def plot_poincare(data):

    plt.scatter(data[:-1], data[1:], s=4, marker='s')

    plt.axis('off')
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")

    fig = plt.gcf()
    fig.canvas.draw()
    array_data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    return array_data


def plot_poincare_seaborn(rr):

    rr_n = rr[:-1]
    rr_n1 = rr[1:]

    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)

    sns.scatterplot(x=rr_n, y=rr_n1)  # color='#51A6D8'

    plt.axis('off')
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")
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

    # if knn is not None:
    #     # pc = RecurrencePlot(dimension=dimension, time_delay=time_delay)
    #     # X_dist = pc.fit_transform(segment)[0]
    #     # X_pc = mask_knn(X_dist, k=knn, policy='cols')
    #     pc = plot_poincare(segment)
    #     # X_pc = mask_knn(pc, k=knn, policy='cols')
    # elif use_clip:
    #     pc = plot_poincare(segment)
    #     # X_pc = pc_norm(pc, threshold='percentage_clipped', percentage=percentage)[0]
    # else:
    #     # pc = RecurrencePlot(dimension=dimension, time_delay=time_delay,
    #     #                     # threshold='percentage_points', percentage=percentage)
    #     #                     threshold='point', percentage=percentage)
    #     # X_pc = pc.fit_transform(segment)[0]
    #     pc = plot_poincare(segment)

    # pc = plot_poincare(segment)
    pc = plot_poincare(segment)

    # if imsize is not None:
    # pc = resize_pc(pc, new_shape=imsize, use_max=True)

    # imageio.imwrite(os.path.join(images_dir, fname), np_to_uint8(
    #     pc), format=suffix, **{"compression": compress})

    imageio.imwrite(os.path.join(images_dir, fname), pc,
                    format=suffix, **{"compression": compress})

    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(pc, cmap=cmap, origin='upper')
        plt.axis('off')
        plt.title('Poincaré Plot for {}'.format(fname), fontsize=14)
        plt.show()
    return fname


def np_to_uint8(X):
    X -= X.min()
    X = (255/X.max())*X
    return X.astype(np.uint8)


def pc_norm(X_dist, threshold=None, percentage=10):
    """Rescale Recurrence Plot after setting nearest-neighbor threshold"""
    n_samples = X_dist.shape[0]    # typically value is 1
    image_size = X_dist.shape[-1]

    assert threshold is not None

    if threshold == 'percentage_points':
        percents = np.percentile(
            np.reshape(X_dist, (n_samples, image_size * image_size)),
            percentage, axis=1
        )
        X_pc = X_dist < percents[:, None, None]
    if threshold == 'percentage_clipped':
        percents = np.percentile(
            np.reshape(X_dist, (n_samples, image_size * image_size)),
            percentage, axis=1
        )
        for i in range(n_samples):
            X_dist[i, X_dist[i] < percents[i]] = percents[i]
            X_dist[i] = percents[i] / X_dist[i]
        X_pc = X_dist**2
    elif threshold == 'percentage_distance':
        percents = percentage / 100 * np.max(X_dist, axis=(1, 2))
        X_pc = X_dist < percents[:, None, None]
    else:
        X_pc = X_dist < threshold
    return X_pc.astype('float64')


def mask_knn(m, k=1, policy='cols'):
    """Creates mask showing knn in each row/column of adjacency matrix"""
    assert policy in ['cols', 'rows']
    mask = np.zeros(m.shape, dtype='bool')
    if policy == 'rows':
        assert m.shape[0] >= k
        vals = np.partition(m, k+1, axis=1)[:, k]  # kth value in each row
        for i in range(m.shape[0]):
            mask[i][m[i] <= vals[i]] = True
    else:
        assert m.shape[1] >= k
        vals = np.partition(m, k+1, axis=0)[k, :]  # kth value in each column
        for i in range(m.shape[1]):
            mask[:, i][m[:, i] <= vals[i]] = True
    return mask


def compute_padding(w, n_align=64):
    """compute required padding for given dimension to naturally align"""
    if w % n_align > 0:
        new_w = ((w // n_align) + 1) * n_align
    else:
        new_w = w
    pad = new_w - w
    pad_l = pad // 2
    pad_r = pad - pad_l
    return new_w, pad_l, pad_r


def align_pc(m, n_align=64):
    """Apply padding to align matrix to given multiple"""
    rows, cols = m.shape

    new_rows, pad_rows_l, pad_rows_r = compute_padding(rows, n_align)
    new_cols, pad_cols_l, pad_cols_r = compute_padding(cols, n_align)
    if rows == new_rows and cols == new_cols:
        padded_m = m
    else:
        padded_m = np.zeros((new_rows, new_cols), dtype=bool)

        if rows == new_rows:
            padded_m[:, pad_cols_l:-pad_cols_r] = a
        elif cols == new_cols:
            padded_m[pad_rows_l:-pad_rows_r, :] = a
        else:
            print('')
            print(padded_m.shape)
            print(padded_m[pad_rows_l:-pad_rows_r,
                  pad_cols_l:-pad_cols_r].shape)
            padded_m[pad_rows_l:-pad_rows_r, pad_cols_l:-pad_cols_r] = m
    return padded_m


def resize_pc(mat, new_shape=64, use_mean=False):
    mat = align_pc(mat, n_align=new_shape)

    rows, cols = mat.shape[0], mat.shape[1]
    downscale_row, downscale_col = rows // new_shape, cols // new_shape
    if use_mean:
        result = np.zeros((new_shape, new_shape))
        for i, ii in enumerate(range(0, rows, downscale_row)):
            for j, jj in enumerate(range(0, cols, downscale_col)):
                result[i, j] = np.mean(
                    mat[ii:ii + downscale_row, jj:jj + downscale_col])
    else:
        result = np.zeros((new_shape, new_shape), dtype=bool)
        for i, ii in enumerate(range(0, rows, downscale_row)):
            for j, jj in enumerate(range(0, cols, downscale_col)):
                result[i, j] = np.max(
                    mat[ii:ii + downscale_row, jj:jj + downscale_col])
    return result
