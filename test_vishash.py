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

from util import compute_features, compute_features_oneimage
from util import normalized_distance, compute_pair_distances
from util import compute_collection_distances, find_duplicates_sym
from util import find_duplicates, mindist, topk


import numpy as np


def test_compute_features():
    path_prefix = 'example/'
    image_names = [
        'USD0873711-20200128-D00000',
        'USD0873712-20200128-D00000',
        'USD0873714-20200128-D00000',
        'corrupt',
        'empty'
    ]
    imglst = [path_prefix + x + '.png' for x in image_names]
    ftlst, gis_list = compute_features(imglst, prunefail=False)
    assert len(ftlst) == len(image_names)
    assert len(gis_list) == len(image_names)
    assert ftlst[3] is None
    assert 0 == np.sum(np.abs(ftlst[4]))

    ftlst, gis_list = compute_features(imglst, prunefail=True)
    assert len(ftlst) == len(image_names) - 2
    assert len(gis_list) == len(image_names) - 2


def test_compute_features_oneimage():
    image_name = 'USD0873711-20200128-D00000'
    file_path = 'example/' + image_name
    image_key = 2
    
    # Check reading different formats of image files
    ftr_png, _ = compute_features_oneimage(image_key, file_path + '.png')
    assert 0 < np.sum(np.abs(ftr_png))
    ftr_jpg, _ = compute_features_oneimage(image_key, file_path + '.jpg')
    assert 0 < np.sum(np.abs(ftr_jpg))
    ftr_tiff, _ = compute_features_oneimage(image_key, file_path + '.tiff')
    assert 0 < np.sum(np.abs(ftr_tiff))

    # Different images produce different signatures
    other_key = 'USD0873714-20200128-D00000'
    file_path = 'example/' + other_key
    ftr_other, _ = compute_features_oneimage(other_key, file_path + '.png')
    diff = np.sum(np.abs(ftr_png - ftr_other))
    assert 0 < diff

    # Different formats of same image produce similar signature
    assert diff > np.sum(np.abs(ftr_png - ftr_jpg))
    assert diff > np.sum(np.abs(ftr_jpg - ftr_tiff))


def test_normalized_distance():
    rng = np.random.default_rng()
    a = rng.integers(low=-3, high=3 + 1, size=5)
    b = rng.integers(low=-3, high=3 + 1, size=5)
    assert 0 == normalized_distance(a, a)
    assert 0 == normalized_distance(b, b)
    assert normalized_distance(a, b) == normalized_distance(b, a)
    assert 1.0 >= normalized_distance(a, b)
    assert 0.0 <= normalized_distance(a, b)


def test_compute_pair_distances():
    rng = np.random.default_rng()
    # 8 sigs of length 6 each
    n_sigs = 8
    dim = 6
    sig_array = rng.integers(low=-3, high=3 + 1, size=(n_sigs, dim))
    all_pairs = compute_pair_distances(sig_array)
    assert (n_sigs, n_sigs) == all_pairs.shape
    all_pairs = compute_pair_distances(sig_array, metric="scaled_euclidean")
    assert (n_sigs, n_sigs) == all_pairs.shape
    all_pairs = compute_pair_distances(sig_array, metric="cosine")
    assert (n_sigs, n_sigs) == all_pairs.shape


def test_compute_collection_distances():
    rng = np.random.default_rng()
    # Two lists of signatures
    # dim must be consistent, but number of sigs can differ
    n_sigs_a = 8
    n_sigs_b = 9
    dim = 7
    sig_array_a = rng.integers(low=-3, high=3 + 1, size=(n_sigs_a, dim))
    sig_array_b = rng.integers(low=-3, high=3 + 1, size=(n_sigs_b, dim))
    dists = compute_collection_distances(sig_array_a, sig_array_b)
    assert (n_sigs_a, n_sigs_b) == dists.shape
    dists2 = compute_collection_distances(sig_array_b, sig_array_a)
    assert (n_sigs_b, n_sigs_a) == dists2.shape
    assert np.sum(dists) - np.sum(dists2) < 10e-6
    dists = compute_collection_distances(
        sig_array_a, sig_array_b, metric="scaled_euclidean"
    )
    assert (n_sigs_a, n_sigs_b) == dists.shape
    dists = compute_collection_distances(sig_array_a, sig_array_b, metric="cosine")
    assert (n_sigs_a, n_sigs_b) == dists.shape


def test_find_duplicates_sym():
    # Symmetric matrix
    mat = np.zeros((5, 5))
    mat[2, 3] = 0.9
    mat[4, 3] = 0.2
    mat = mat + np.transpose(mat)
    assert (0, 1) in np.transpose(find_duplicates_sym(mat, 0.1))
    assert np.sum(mat[find_duplicates_sym(mat, 0.1)]) == 0
    assert np.sum(mat[find_duplicates_sym(mat, 0.5)]) == 0.2
    assert np.sum(mat[find_duplicates_sym(mat, 1)]) == 0.2 + 0.9


def test_find_duplicates():
    mat = np.zeros((4, 6))
    mat[3, 2] = 0.9
    mat[3, 4] = 0.2
    assert (0, 1) in np.transpose(find_duplicates(mat, 0.1))
    assert np.sum(mat[find_duplicates(mat, 0.1)]) == 0
    assert np.sum(mat[find_duplicates(mat, 0.5)]) == 0.2
    assert np.sum(mat[find_duplicates(mat, 1)]) == 0.2 + 0.9


def test_mindist():
    mat = np.zeros((4, 4))
    n_pairs = int(mat.shape[0] * (mat.shape[1] - 1) / 2)
    mat[2, 3] = 0.9
    mat[3, 1] = 0.2
    mat = mat + np.transpose(mat)

    mins = mindist(mat, n=3)
    assert len(mins[0]) == 3
    id_list = list(zip(*mins))
    assert mat[id_list[0]] == 0

    mins = mindist(mat, 50)
    assert n_pairs == len(mins[1])
    id_list = list(zip(*mins))
    assert mat[id_list[0]] < mat[id_list[n_pairs-1]]


def test_topk():
    mat = np.zeros((4, 6))
    n_images, max_k = mat.shape
    mat[3, 2] = 0.9
    mat[3, 4] = 0.2
    matches = topk(mat)
    assert matches.shape == mat.shape
    assert mat[3, matches[3, 0]] < mat[3, matches[3, max_k-1]]
    assert mat[3, matches[3, 0]] < mat[3, matches[3, max_k-2]]
    assert mat[3, matches[3, 0]] == mat[3, matches[3, 1]]

    matches = topk(mat, k=2)
    assert matches.shape == (mat.shape[0], 2)

