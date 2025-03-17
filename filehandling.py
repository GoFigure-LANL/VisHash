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

import numpy as np
import numbers
import csv
import json
import glob


# Writing and reading distance matrix


def write_distance_mat(filename, path, data):
    np.savetxt(path + "/" + filename, data)


def read_distance_mat(postfix, path):
    dist_mat = np.loadtxt(path + "/mat_" + postfix + ".csv")
    return dist_mat


# Utilities for filenames


def clean_filename(filename):
    """Remove leading path and ending carriage return from filename."""
    return filename.split("/")[-1].strip()


def write_filenames(filename, path, file_list):
    # Save a list of the filenames with index
    with open(path + "/" + filename, "w", encoding="utf-8") as outfile:
        for i, filename in enumerate(file_list):
            outfile.write(str(i) + ", " + filename + "\n")


def read_filenames(postfix, path):
    fnm = []
    with open(path + "/filenames_" + postfix + ".csv", encoding="utf-8") as infile:
        for line in infile:
            words = line.split(", ")
            if words[0].isnumeric():
                fnm.append(clean_filename(words[1]))
    return fnm


def read_filename_lookup(postfix, path):
    with open(path + "/filenames_" + postfix + ".json", encoding="utf-8") as infile:
        fnm_lookup = json.load(infile)
    return fnm_lookup


def write_filename_lookup(postfix, path, data):
    with open(
        path + "/filenames_" + postfix + ".json", "w", encoding="utf-8"
    ) as outfile:
        json.dump(data, outfile)


def scan_files(pth, N):
    """
    Gather names of all files in given path.

    Returns: fnm, a list of strings representing the full path filename
    """
    fnm = glob.glob(pth + "/*")
    if N is not None:
        fnm = fnm[0:N]

    return fnm


def read_filenames_from_file(pth, input_file, N):
    """
    Read given input_file to gather names of image files. Preprend pth to each
    filename from input_file.

    Returns: fnm, a list of strings representing the full path filename
    """
    with open(input_file, encoding="utf-8") as f:
        fnm = [pth + "/" + x.strip() for x in f.readlines()]
    if N is not None:
        fnm = fnm[0:N]

    return fnm


# Read in mindists file
def read_mindists(postfix, path):
    with open(path + "/mindist_" + postfix + ".csv", encoding="utf-8") as infile:
        mindists = {"k": [], "fileA": [], "fileB": [], "dist": []}
        reader = csv.DictReader(infile)
        for row in reader:
            mindists["k"].append(row["order"])
            mindists["fileA"].append(row["fileA"])
            mindists["fileB"].append(row["fileB"])
            mindists["dist"].append(row["dist"])
    return mindists


def write_mindists(postfix, path, file_pairs, dist_list):
    with open(path + "/mindist_" + postfix + ".csv", "w", encoding="utf-8") as outfile:
        fieldnames = ["order", "fileA", "fileB", "dist"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(file_pairs)):
            writer.writerow(
                {
                    "order": i,
                    "fileA": file_pairs[i][0],
                    "fileB": file_pairs[i][1],
                    "dist": dist_list[i],
                }
            )


def save_matchlist(midx, dst, filenames, postfix):
    """
    For all images in match list, write a file with the filenames of the images
    and the distance. The output is written in the same order as the given
    midx.
    """
    # Check that there is actually a list to construct
    if isinstance(midx[0], (list, np.ndarray)):
        n_pairs = len(midx[0])
        file_pairs = [
            (filenames[midx[0][i]], filenames[midx[1][i]]) for i in range(n_pairs)
        ]
        dist_list = [dst[midx[0][i], midx[1][i]] for i in range(n_pairs)]

    # Only one match
    elif isinstance(midx[0], numbers.Number):
        file_pairs = [(filenames[midx[0]], filenames[midx[1]])]
        dist_list = [dst[midx[0], midx[1]]]

    # No matches
    else:
        file_pairs = []
        dist_list = []

    write_mindists(postfix, "./", file_pairs, dist_list)


# Read in all matches
def read_matches(postfix, path):
    with open(path + "/matches_" + postfix + ".csv", encoding="utf-8") as infile:
        mindists = {"k": [], "fileA": [], "fileB": [], "dist": []}
        reader = csv.DictReader(infile)
        for row in reader:
            mindists["k"].append(row["order"])
            mindists["fileA"].append(row["fileA"])
            mindists["fileB"].append(row["fileB"])
            mindists["dist"].append(row["dist"])
    return mindists


def write_matches(postfix, path, file_pairs, dist_list):
    with open(path + "/matches_" + postfix + ".csv", "w", encoding="utf-8") as outfile:
        fieldnames = ["order", "fileA", "fileB", "dist"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(file_pairs)):
            writer.writerow(
                {
                    "order": i,
                    "fileA": file_pairs[i][0],
                    "fileB": file_pairs[i][1],
                    "dist": dist_list[i],
                }
            )


# top-k closest matches
def read_topk(postfix, path):
    with open(path + "/topk_" + postfix + ".json", encoding="utf-8") as infile:
        query_dict = json.load(infile)
    return query_dict


def write_topk(postfix, path, query_dict):
    """
    query_dict is of form query_filename:(match_filename_list, dist_list);
    where match_filename_list and dist_list are of the same length.
    """
    with open(path + "/topk_" + postfix + ".json", "w", encoding="utf-8") as outfile:
        json.dump(query_dict, outfile)


# false positive list
def write_fp_list(postfix, path, fp_list, mindists):
    with open(path + "/fp_list_" + postfix + ".csv", "w", encoding="utf-8") as outfile:
        fieldnames = ["fp_count", "fileA", "fileB", "num_retrieved"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(fp_list)):
            num_retrieved = fp_list[i]
            writer.writerow(
                {
                    "fp_count": i,
                    "fileA": mindists["fileA"][num_retrieved],
                    "fileB": mindists["fileB"][num_retrieved],
                    "num_retrieved": num_retrieved,
                }
            )
