### ©2020. Triad National Security, LLC. All rights reserved.
### This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

import argparse
import glob
import numpy as np
import time
import matplotlib.pyplot as plt

import util
import filehandling



def compute_sigs(fnm, prfx):
    """
    Given the list of filenames, compute VisHash for each image. Two output files are
    written:
    filenames_[prfx].csv: List of filenames of images processed
    signatures_[prfx].npy: Numpy array of hashes, in the same order as the filenames list
    """
    print(time.asctime(), ' Computing signatures')
    start = time.time()
    (ftr, grid_list) = util.compute_features(fnm)
    end = time.time()
    print(time.asctime(), ' Done Computing signatures in ', end-start, ' seconds')
    print('There are ', str(len(fnm)), ' images', flush=True)
    np.save('signatures_' + prfx, ftr)
    
    # Save a list of the filenames with index
    out_file = 'filenames_' + prfx + '.csv'
    filehandling.write_filenames(out_file, './', fnm)

    return ftr


def calc_distances(ftr, prfx=None):
    """
    Compute all-unordered-pairs distances on the given list of hashes (ftr). If prfx is 
    given, save the full dst matrix as mat_[prfx].npy (not recommended, as this can be
    a very large file).
    """
    print(time.asctime(), ' Computing distances')
    start = time.time()
    dst = util.compute_pair_distances(ftr)
    end = time.time()
    print(time.asctime(), ' Done Computing distances in ', end-start, ' seconds', flush=True)

    # Only save if requested (this can be a very large file)
    if prfx is not None:
        np.save('mat_' + prfx, dst)

    return dst


def get_min_distances(dst, n_matches):
    """
    For the given distance matrix, return a tuple of two lists of indices such that
    dst[midx[0][i], midx[1][i]] are the minimum-distance values in dst, for i<n_matches.
    Returns: midx of the form ([indexA], [indexB]) 
    """
    print(time.asctime(), ' Sorting distances')
    start = time.time()
    midx = util.mindist(dst, n=n_matches)
    end = time.time()
    print(time.asctime(), ' Done Sorting distances in ', end-start, ' seconds', flush=True)
    return midx





def get_matches(dst, threshold):
    """
    Find all pairs in dst with dst[i][j] < threshold

    Returns: dup_dict dictionary of pairs of images
    """
    print(time.asctime(), ' Filtering and sorting distances')
    start = time.time()
    dup_dict = util.find_duplicates(dst, threshold)
    n_matches = len(dup_dict)
    end = time.time()
    print(time.asctime(), ' Done Filtering distances for ', n_matches, ' matches in ', end-start, ' seconds', flush=True)
    return dup_dict
    


def get_min_distances_per_query(dst, n_matches_per_query):
    """
    For every row of dst, find the n_matches_per_query smallest distances.
    
    Returns: query_matches, a list of lists of matches per query
    """
    print(time.asctime(), ' Sorting distances')
    start = time.time()
    query_matches = []
    for i in range(dst.shape[0]):
        midx = util.mindist(dst[i,:], n=n_matches_per_query)
        query_matches.append(midx)
    end = time.time()
    print(time.asctime(), ' Done Sorting distances in ', end-start, ' seconds', flush=True)
    return query_matches


###
### Main and commandline arguments
###
def get_args():
    parser = argparse.ArgumentParser(description='Calculate VisHash for all images in given directory.')
    parser.add_argument('--dataname', type=str, help='The name of the dataset (for storing output files)')
    parser.add_argument('--image_path', type=str, default='./',
                        help='Path to where images are stored [./]')
    parser.add_argument('--filenames', type=str, default=None,
                        help='Optional name of file containing filenames to be processed instead of processing the whole directory [None]')
    parser.add_argument('--num_images', type=int, default=None,
                        help='If specified limit number of images to num_images (usually for testing purposes)')
    parser.add_argument('--compute_distances', action="store_true", default=False,
                        help='If flag present, calculate all-pairs distances [False]')
    parser.add_argument('--n_matches', type=int, default=1000,
                        help='If specified limit number of matches n_matches (strongly recommended)')
    args = parser.parse_args()
    return(args)



def main():
    args = get_args()
    postfix = args.dataname

    if args.filenames is None:
        filenames = filehandling.scan_files(args.image_path, args.num_images)
    else:
        filenames = filehandling.read_filenames_from_file(args.image_path, args.filenames, args.num_images)
    sigs = compute_sigs(filenames, postfix)

    if args.compute_distances:
        dist_mat = calc_distances(sigs)
        midx = get_min_distances(dist_mat, args.n_matches)
        filehandling.save_matchlist(midx, dist_mat, filenames, postfix)
        

if __name__ == '__main__':
    main()

