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

# Script to generate images in VisHash publication (Table 1).
# Usage: python show_sig_step.py path_to_folder_of_images

import argparse
import numpy as np
import copy

import util
import filehandling


# Main and commandline arguments


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate VisHash for all images in given directory."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./",
        help="Path to where images are stored [./]",
    )
    parser.add_argument(
        "--filenames",
        type=str,
        default=None,
        help=(
            "Optional name of file containing filenames to be processed ",
            "instead of processing the whole directory, or to ensure order ",
            "of filenames [None]",
        ),
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help=(
            "If specified limit number of images to num_images ",
            "(usually for testing purposes)",
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Gather the filenames
    if args.filenames is None:
        filenames = filehandling.scan_files(args.image_path, args.num_images)
    else:
        filenames = filehandling.read_filenames_from_file(
            args.image_path, args.filenames, args.num_images
        )

    image_names = [x.split("/")[-1] for x in filenames]

    # Calculate the hash for each image, and save the intermediate calculations
    (_, gis_list) = util.compute_features(filenames)

    # For hash-difference image, scale colors to maximum possible distance
    # for given reference image (double the max absolute value)
    max_color = 2 * np.max(np.abs(gis_list[0].level_vec))

    # Loop through the hashes and dispay each calculation step
    for i, gis in enumerate(gis_list):
        grid_img = util.get_grid_mask(gis)
        grid_img.save("grid_" + image_names[i])

        filename = "mean_" + image_names[i]
        util.img_to_meangray(gis, filename=filename)

        filename = "diff_left_" + image_names[i]
        util.img_of_differentials(gis, filename=filename, neighbors="left_only")
        filename = "diff_up_" + image_names[i]
        util.img_of_differentials(gis, filename=filename, neighbors="up_only")
        filename = "diff_" + image_names[i]
        util.img_of_differentials(gis, filename=filename)

        filename = "level_left_" + image_names[i]
        util.img_of_levels(gis, filename=filename, neighbors="left_only")
        filename = "level_up_" + image_names[i]
        util.img_of_levels(gis, filename=filename, neighbors="up_only")
        filename = "level_" + image_names[i]
        util.img_of_levels(gis, filename=filename)

        gis_diff = copy.deepcopy(gis)
        gis_diff.diff_vec = gis_diff.level_vec - gis_list[0].level_vec
        filename = "sigdiff_" + image_names[i]
        util.img_of_differentials(gis_diff, filename=filename, max_color=max_color)


if __name__ == "__main__":
    main()
