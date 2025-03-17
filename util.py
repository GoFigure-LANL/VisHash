# ©2020. Triad National Security, LLC. All rights reserved.
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do
# so.

# Authors: Brendt Wohlberg <brendt@lanl.gov> and Diane Oyen

import numpy as np
from PIL import Image, ImageDraw
import warnings

import matplotlib.pyplot as plt
import scipy.spatial

from vishash import ImageSignature

warnings.filterwarnings("ignore", ".*output shape of zoom.*")


# Helper functions to read images, analyze, and output VisHash
# signatures/hashes, distances, and matches.


def compute_features(imglst, prunefail=True):

    ftlst = []
    plst = []
    gis_list = []
    for k, f in enumerate(imglst):
        # Open file and calculate signature
        ftr, gis = compute_features_oneimage(k, f)

        # check whether signature succeeded
        fail = True if ftr is None else False

        if ftr is not None and np.sum(np.abs(ftr)) == 0:
            fail = True
            print("Zero signature for image (%d) %s" % (k, f))

        # Append to lists if it is succesful, or we are not pruning
        if not fail or not prunefail:
            ftlst.append(ftr)
            gis_list.append(gis)

        else:
            # Otherwise, don't append it. Mark it for pruning.
            plst.append(k)

    # If pruning, remove names of failed signatures from imglst
    for k in plst[::-1]:
        del imglst[k]

    return (ftlst, gis_list)


def compute_features_oneimage(k, f):
    ftr = None
    try:
        try:
            gis = ImageSignature()
            ftr = gis.generate_signature(f)
        except Exception as e:
            # works for TIFF, why not webp? What about svg?
            print(
                ("Re-trying image signature calculation for image ", "(%d) %s" % (k, f))
            )
            print(e)
            im = Image.open(f)
            gis = ImageSignature()
            ftr = gis.generate_signature(im.tobitmap(), bytestream=True)
    except Exception as e:
        ftr = None
        print("Error computing signature for image (%d) %s" % (k, f))
        print(e)
    return ftr, gis


# Distance calculations


def normalized_distance(a, b):
    """Compute normalize distance between two points.
    Computes || b - a || / ( ||b|| + ||a||)
    Args:
        a (numpy.ndarray): array of length p
        b (numpy.ndarray): array of length p
    Returns:
        normalized distance between signatures (float)
    """
    b = np.array(b).astype(float)
    a = np.array(a).astype(float)
    norm_diff = np.linalg.norm(b - a)
    norm1 = np.linalg.norm(b)
    norm2 = np.linalg.norm(a)
    return norm_diff / (norm1 + norm2)


def compute_pair_distances(sig_array, metric="scaled_euclidean"):
    """Compute distances between feature vectors of all pairs.

    Args:
        sig_array: NxP array of N P-dimensional signatures/hashes/feature
            vectors
    Returns:
        NxN symmetric matrix (float) of distances
    """

    sig_array = np.array(sig_array)

    if metric == "scaled_euclidean":
        sig_norms = np.linalg.norm(sig_array, axis=1)
        n_sigs = sig_array.shape[0]
        sums = np.array(sig_norms[:, None] + sig_norms[None, :])[
            np.triu_indices(n_sigs, k=1)
        ]
        dists = scipy.spatial.distance.pdist(sig_array, metric="euclidean")
        dists = dists / sums

    elif metric == "cosine":
        dists = scipy.spatial.distance.pdist(sig_array, metric="cosine")

    return scipy.spatial.distance.squareform(dists)


def compute_collection_distances(sig_array_a, sig_array_b, metric="scaled_euclidean"):
    """Compute distances between all feature vectors from sig_array_a and
    sig_array_b.

    Args:
        sig_array_a: NxP array of N P-dimensional signatures/hashes/feature
            vectors
        sig_array_b: MxP array of M P-dimensional signatures/hashes/feature
            vectors
    Returns:
        NxM matrix (float) of distances
    """
    dists = None

    if metric == "scaled_euclidean":
        sig_norms_a = np.linalg.norm(sig_array_a, axis=1)
        sig_norms_b = np.linalg.norm(sig_array_b, axis=1)
        sums = np.array(sig_norms_a[:, None] + sig_norms_b[None, :])
        dists = scipy.spatial.distance.cdist(
            sig_array_a, sig_array_b, metric="euclidean"
        )
        dists = dists / sums

    elif metric == "cosine":
        dists = scipy.spatial.distance.cdist(sig_array_a, sig_array_b, metric="cosine")

    return dists


# Threshold distances to find duplicates, or top matches


def find_duplicates_sym(dist_mat, threshold):
    # Mask removes diagonal and ordered pairs by setting them >= 1
    mask = np.tril(np.ones_like(dist_mat)) + dist_mat
    matches = np.nonzero(mask <= threshold)
    return matches


def find_duplicates(dist_mat, threshold):
    return np.nonzero(dist_mat <= threshold)


def mindist(dst, n=1):
    """
    Get minimum off-diagonal distance in distance matrix. Return at most
    the number of valid unordered pairs.
    """
    # Calculate the maximum number of unordered pairs
    n_images = dst.shape[0]
    max_pairs = n_images * (n_images - 1) / 2
    n = int(min(max_pairs, n))

    sym = np.sum(np.abs(dst - dst.T)) == 0.0
    dst = dst.copy()
    dst[np.eye(dst.shape[0], dtype=bool)] = np.inf
    if sym:
        dst[np.triu_indices(dst.shape[0])] = np.inf
    if n == 1:
        idx = np.unravel_index(dst.argmin(), dst.shape)
    else:
        # For very large distance matrices, partition to
        # identify top n matches before
        # doing a full sort on those matches

        # flatten to a vector
        mask = np.tril_indices(n_images, k=-1)
        dst_flat = np.array(dst[mask])
        # partition first n matches
        partition = np.argpartition(dst_flat, n - 1)
        subset = dst_flat[partition][0:n]
        # full sort on n matches
        srt = np.argsort(subset)
        # return indices in original space
        id0 = np.array([mask[0][i] for i in partition[srt]])
        id1 = np.array([mask[1][i] for i in partition[srt]])
        idx = (id0, id1)

    return idx


def topk(dst, k=10):
    """
    For each row of dst, find the lowest k distances.
    """
    # get number of queries
    n_images = dst.shape[0]
    max_k = dst.shape[1]

    # If symmetric, need to mask out the diagonal
    sym = False if not n_images == max_k else \
        np.sum(np.abs(dst - dst.T)) < 1e-5
    if sym:
        dst = dst.copy()
        dst[np.eye(dst.shape[0], dtype=bool)] = np.inf
        max_k = max_k - 1  # can't match itself

    # Adjust k if it is larger than the number of possible matches
    k = min(k, max_k)

    idx = np.zeros((n_images, k), dtype=np.int32)
    # Loop through the rows
    for i, row in enumerate(dst):
        # For very large distance matrices, partition to
        # identify top k matches before
        # doing a full sort on those matches

        # partition first n matches
        partition = np.argpartition(row, k - 1)
        subset = row[partition][0:k]
        # full sort on k matches
        srt = np.argsort(subset)
        # return indices in original space
        idx[i, :] = partition[srt]

    return idx


# Visualization


def impairshow(img0, img1, title):
    """
    Display a pair of images by creating a new image
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    fig.suptitle(title, fontsize=14)
    # Display first image
    ax[0].imshow(img0, cmap="gray")
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    # Display second image
    ax[1].imshow(img1, cmap="gray")
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    fig.show()

    return fig, ax


def get_grid_mask(gis):
    """
    Create an image of the grid defined by gis.
    """
    arr = gis.im_array
    p_width = np.int(gis.patch_width / 2)
    p_height = np.int(gis.patch_height / 2)

    # Setup drawing image
    img = Image.fromarray(arr)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Get grid lines
    for x in gis.x_coords:
        for y in gis.y_coords:
            # x and y are opposite in Image space versus numpy space
            rect_coords = [(y - p_height, x - p_width), (y + p_height, x + p_width)]
            draw.rectangle(rect_coords, fill=None, outline="red")

    del draw  # not sure why this is in every example
    return img


def img_to_meangray(gis, filename):
    """
    Create and save an image that is a block of gray for every patch in the
    VisHash grid.
    """
    # need high enough resolution image so it renders correctly
    square_width = 100

    arr = np.zeros((square_width * len(gis.x_coords), square_width * len(gis.y_coords)))
    start_x = 0
    for i in range(len(gis.x_coords)):
        end_x = start_x + square_width
        start_y = 0
        for j in range(len(gis.y_coords)):
            end_y = start_y + square_width
            arr[start_x:end_x, start_y:end_y] = gis.gray_level_matrix[i, j]
            start_y = end_y  # get ready for next row
        start_x = end_x  # get ready for next column

    plt.imsave(filename, arr, cmap="gray")
    return arr


def split_neighbor_vec(vector, n):
    """
    Takes a vector and splits into two parts returning two numpy matrices of
    dimension n x (n-1) for the left neighbors, and (n-1) x n for the upper
    neighbors.
    """
    split_point = int(len(vector) / 2)
    left_neighbors = vector[:split_point]
    # should be n x (n-1) matrix
    left_neighbors = np.reshape(left_neighbors, (n, n - 1))

    up_neighbors = vector[split_point:]
    # should be (n-1) x n matrix
    up_neighbors = np.reshape(up_neighbors, (n - 1, n))

    return (left_neighbors, up_neighbors)


def split_square_to_triangle(square_width):
    """
    For the given square_width, return the matrix indices associated with the
    upper triangle and with the left triangle as
    (up_xs, up_ys, left_xs, left_ys).
    Used to display hash values as an image.
    """
    up_tri = np.ones((square_width, square_width))
    up_tri = np.triu(up_tri, k=1)
    up_xs, up_ys = np.nonzero(up_tri)

    left_tri = np.ones((square_width, square_width))
    left_tri = np.tril(left_tri, k=-1)
    left_xs, left_ys = np.nonzero(left_tri)

    return (up_xs, up_ys, left_xs, left_ys)


def img_of_differentials(gis, filename=None, neighbors=None, max_color=None):
    """
    Create an image of the mean gray level differences (two values per grid
    point). If max_color given, will use |max_color| to scale the colors,
    otherwise will calculate based on max absolute value given in gis.
    """
    # get differentials and re-instate their shape
    left_neighbors, up_neighbors = split_neighbor_vec(gis.diff_vec, gis.n)

    # Split each grid into an upper-right triangle for the upper_neighbors
    # and lower-left triangle for the left_neighbors
    square_width = 100
    arr = np.zeros((square_width * gis.n, square_width * gis.n))
    # get triu and tril (minus diag) for first patch
    up_xs, up_ys, left_xs, left_ys = split_square_to_triangle(square_width - 1)

    # Add the appropriate patch width to each loop iteration
    x_shift = 0
    for i in range(len(gis.x_coords)):
        y_shift = 0
        for j in range(len(gis.y_coords)):
            # first row has no up_neighbors
            if (not neighbors == "left_only") and i > 0:
                arr[up_xs + x_shift, up_ys + y_shift] = up_neighbors[i - 1, j]
            # first column has no left_neighbors
            if (not neighbors == "up_only") and j > 0:
                arr[left_xs + x_shift, left_ys + y_shift] = left_neighbors[i, j - 1]
            y_shift += square_width
        x_shift += square_width

    # Color-code for positive and negative values
    # blue are positive, red are negative
    if max_color is None:
        max_color = np.max(np.abs(arr))
    else:
        max_color = np.abs(max_color)
    plt.imsave(filename, arr, cmap="seismic", vmin=-max_color, vmax=max_color)
    return arr


def img_of_levels(gis, filename=None, neighbors=None):
    """
    Create an image of the levels (thresholded, normalized, and binned) of
    differences between neighbors (two discrete values per grid point).
    """
    # get levels and re-instate their shape
    left_neighbors, up_neighbors = split_neighbor_vec(gis.level_vec, gis.n)

    # Split each grid into an upper-right triangle for the upper_neighbors
    # and lower-left triangle for the left_neighbors
    square_width = 100
    arr = np.zeros((square_width * gis.n, square_width * gis.n))
    # get triu and tril (minus diag) for first patch
    up_xs, up_ys, left_xs, left_ys = split_square_to_triangle(square_width - 1)

    # Add the appropriate patch width to each loop iteration
    x_shift = 0
    for i in range(len(gis.x_coords)):
        y_shift = 0
        for j in range(len(gis.y_coords)):
            # first row has no up_neighbors
            if (not neighbors == "left_only") and i > 0:
                arr[up_xs + x_shift, up_ys + y_shift] = up_neighbors[i - 1, j]
            # first column has no left_neighbors
            if (not neighbors == "up_only") and j > 0:
                arr[left_xs + x_shift, left_ys + y_shift] = left_neighbors[i, j - 1]
            y_shift += square_width
        x_shift += square_width

    # Color-code for positive and negative values
    # blue are positive, red are negative
    plt.imsave(filename, arr, cmap="seismic")
    return arr
