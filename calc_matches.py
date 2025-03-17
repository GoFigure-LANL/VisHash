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

import argparse
import numpy as np
import time

import util
import filehandling


def calc_distances(ftr):
    start = time.time()
    dst = util.compute_pair_distances(ftr)
    end = time.time()
    print(
        time.asctime(),
        " Done Computing distances in ",
        end - start,
        " seconds",
        flush=True,
    )

    return dst


def get_matches_sym(dst, threshold):
    """
    Returns all matching unordered pairs below given threshold. dst assumed
    to be symmetric.
    """
    start = time.time()
    midx = util.find_duplicates_sym(dst, threshold)
    n_matches = len(midx[0])
    end = time.time()
    print(
        time.asctime(),
        " Done Filtering distances for ",
        n_matches,
        " matches in ",
        end - start,
        " seconds",
        flush=True,
    )
    return midx


def get_matches(dst, threshold):
    """
    Returns all matching pairs below given threshold. dst need not be square.
    """
    start = time.time()
    midx = util.find_duplicates(dst, threshold)
    n_matches = len(midx[0])
    end = time.time()
    print(
        time.asctime(),
        " Done Filtering distances for ",
        n_matches,
        " matches in ",
        end - start,
        " seconds",
        flush=True,
    )
    return midx


def save_matchlist(midx, dst, filenames, postfix):
    """
    Write a file that lists every match as:
    index,filenameA,filenameB,distance
    """
    n_pairs = len(midx[0])
    file_pairs = [
        (filenames[midx[0][i]], filenames[midx[1][i]]) for i in range(n_pairs)
    ]
    dist_list = [dst[midx[0][i], midx[1][i]] for i in range(n_pairs)]
    filehandling.write_matches(postfix, "./", file_pairs, dist_list)


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate distances between all-pairs of given ",
            "signature list and store matches below given threshold.",
        )
    )
    parser.add_argument(
        "--postfix",
        type=str,
        help=(
            "filenames_[postfix].csv and signatures_[postfix].npy will be ",
            "read while matches_[postfix].csv will be written",
        ),
    )
    parser.add_argument(
        "--path", type=str, default="./", help="Path to input files [./]"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help=(
            "Distance threshold, such that matches will only be stored ",
            "for distance < threshold [0.3]",
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    postfix = args.postfix

    # Read signatures and filename list
    filenames = filehandling.read_filenames(postfix, args.path)
    sigs = np.load(args.path + "/signatures_" + postfix + ".npy")

    # Compute all-pairs distances
    dist_mat = calc_distances(sigs)
    match_idx = get_matches_sym(dist_mat, args.threshold)
    save_matchlist(match_idx, dist_mat, filenames, postfix)


if __name__ == "__main__":
    main()
