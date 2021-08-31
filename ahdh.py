### ©2020. Triad National Security, LLC. All rights reserved.
### This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

### Python implementation of Adaptive Hierarchical Density Histogram (AHDH)
###
### Diane Oyen


# Standard packages
import argparse
import copy
import numpy as np
import glob
import time

# Image packages
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
import skimage



class AdaptiveHistogram(object):
    """ Image signature generator using the Adaptive Hierarchical 
    Density Histogram (AHDH) method.

    Args:
        n_levels (Optional[int]): number of hierarchical levels. (default 10)
        n_nonquant_levels (Optional[int]): lowest level which will use quantized relative levels (default 2)
    """
    
    def __init__(self, n_levels=10, n_nonquant_levels=2, output_dir='./',
                 save_all_images=False):
        """ Initialize the signature generator
        """
        self.im_array = None

        assert type(n_levels) is int, 'n_levels should be an integer > 0'
        assert n_levels > 0, 'n_levels should be greater than 0 (%r given)' % n_levels
        self.n_levels = n_levels

        assert type(n_nonquant_levels) is int, 'n_nonquant_levels should be an integer'
        assert n_nonquant_levels >= 0, 'n_nonquant_levels should be at least 0 (%r given)' % n_quantized_level
        self.n_nonquant_levels = n_nonquant_levels

        # Make sure output_dir has the trailing slash (but don't make it root)
        if output_dir == '':
            self.output_dir = './'
        else:
            self.output_dir = output_dir + '/' 

        self.save_all_images = save_all_images
        

    def generate_signature(self, fname):
        """ Generates an image signature. Paper mentions simple noise reduction 
        and coordinate normalization but does not describe these steps, and so
        they are not included in this code. The returned image signature contains
        (4/3)*(4^(n_nonquant_levels) - 1) + 15*(n_levels - n_nonquant_levels) real values.

        Args:
            fname (string): image path

        Returns:
            The image signature (numpy vector)
        """

        # Load image as array of binary values
        self.load_img_as_binary(fname)
        if self.im_array is None:
            return None


        # Recurse through hierarchical regions starting at level=0
        self.features = []
        regions = [[self.im_array]]
        n_pixels = np.array([len(np.where(self.im_array)[0])])
        areas = None # Not needed until quantized layers
        for level in range(self.n_levels):
            regions, n_pixels, areas = self.ahdh_extraction(regions, n_pixels, areas, level)

        return self.features

    
    ### I/O methods ###

    def load_img_as_binary(self, fname):
        """
        Read in the given image and convert to binary. Return ndarray. If read fails,
        return None value.
        """
        # Read the image into an array as gray-value
        try:
            # imread produces 2-d array with black = 0, white = 255, uint8
            self.im_array = imread(fname, as_gray=True)
        except:
            print("Failed to open image " + fname)
            return

        # Keep the image's filename if needed
        self.im_name = fname.split('/')[-1] # strip leading path
        self.im_name = self.im_name.split('.')[0] # strip extension

        # Threshold to binary
        threshold = threshold_otsu(self.im_array)
        # "less than' inverts the grascale so that
        # black (0 from imread) is foreground (True in binary)
        self.im_array = self.im_array <= threshold
        if self.save_all_images:
            self.save_image('bw_', self.im_array)


    def save_image(self, prefix, bin_array):
        """   
        Takes a binary array, inverts black/white, then saves to the given filename. 
        """
        filename = self.output_dir + '/' + prefix + self.im_name + '.png'
        imsave(filename, skimage.img_as_ubyte(np.logical_not(bin_array)),
               check_contrast=False)
        return


    def save_partition_images(self, prefix, regions, centroids):
        # Could be a better way of doing this - probably lines of subregion edges,
        # and accumulate on the full image rather than save just the region
        # Create cyan lines at the centroid
        for i, region in enumerate(regions):
            # Make sure it will be a valid image
            if region.size < 1:
                continue
            im_color = np.ones(region.shape, dtype=np.uint8) * 255 # background
            im_color[region] = 0 # foreground
            im_color = skimage.color.gray2rgb(im_color)
            cyan = [0, 255, 255]
            this_row, this_col = centroids[i]
            im_color[:, this_col] = cyan
            im_color[this_row, :] = cyan
            filename = self.output_dir + '/' + prefix + str(self.level) + '_' + str(i) + '_' + self.im_name + '.png'
            imsave(filename, im_color, check_contrast=False)
    


    ### AHDH Calculation Methods ###
            
    def ahdh_extraction(self, regions, n_pixels, areas, level):
        """
        For the given level, calculate the feature vector of the given regions. Returns
        a vector of length 4^(level + 1) of densities, unless this is a quantized relative
        density level, in which case the vector is of length 15 representing a histogram.

        Args:
            regions (list of length 4^level of variable-sized ndarrays): Binary images
            n_pixels (4^level x 1 ndarray): Number of foreground pixels per region
            areas (4^level x 1 ndarray or None): Number of all pixels per region, 
                 or None if not yet calculated
            level (int): Which layer of the hierarchy, starts at 0

        Returns:
            features (4^(level+1)x1 ndarray or 15x1 ndarray): Density features for this level
        """
        # Need to ensure there is some value for new_areas, even if it is None
        new_areas = areas
        if areas is not None:
            # Flatten areas array before starting level
            areas = areas.flatten()

        # Get the subregions
        regions_flat = [item for sublist in regions for item in sublist]
        all_subregions = [self.get_subregions(x) for x in regions_flat]

        # Count foreground pixels
        new_n_pixels = np.array([self.count_foreground(x) for x in all_subregions])

        # Calculate densities
        densities = self.calc_densities(new_n_pixels, n_pixels)
        
        # Check if this should be quantized relative densities
        if level >= self.n_nonquant_levels:
            # Need region-level areas if this is the first quant level
            if areas is None:
                areas = np.array([x.size for x in regions_flat])

            # Get areas of subregions
            new_areas = np.array([self.calc_areas(x) for x in all_subregions])

            # Quantize relative densities
            scale_densities = densities * areas[:, None]
            rel_densities = scale_densities >= new_areas
            # Make histogram of quantized code words
            region_features = self.count_codes(rel_densities)
        else:
            region_features = densities.flatten()

        self.features.extend(region_features)
            
            
        # Recurse on the subregions (actually using loop rather than recursion)
        return all_subregions, new_n_pixels.flatten(), new_areas


    def find_centroid(self, region):
        """
        Calculates the centroid (row, col) of the image based
        on locations of foreground pixels
        """
        rows, cols = np.where(region)
        # If empty, just use the middle as the centroid
        if (len(rows) < 1):
            centroid_row, centroid_col =  [int(x/2) for x in region.shape]
        else:
            centroid_row = int(np.mean(rows))
            centroid_col = int(np.mean(cols))
        
        return (centroid_row, centroid_col)


    def get_subregions(self, region):
        """
        For the given region, find the centroid and return the slices associated
        with the four new regions based on the centroid.
        """
        row, col = self.find_centroid(region)
        region1 = region[:row, :col]
        region2 = region[:row, col:]
        region3 = region[row:, :col]
        region4 = region[row:, col:]
        
        return [region1, region2, region3, region4]


    def count_foreground(self, subregions):
        """
        For the given list of subregions, count the number of foreground pixels.
        Returns a numpy vector of same length as the number of subregions.
        """
        n_foreground_pixels = np.array([len(np.where(x)[0]) for x in subregions])
        return n_foreground_pixels

    
    def calc_densities(self, n_subregion, n_region):
        """
        Calculate the [subregion number of foreground pixels] / [region number of
        foreground pixels] for each subregion. In the case that the region has 0
        foreground pixels, the density for all of the associated subregions is 1.

        Args
            n_subregion (4^level x 4 ndarray): Number of foreground pixels per subregion,
                where each row is a set of subregions sharing the same parent region
            n_region (4^level x 1 ndarray): Number of foreground pixels per region

        Return
            densities (4^level x 4 ndarray)
        """
        # Div by zero is handled, so don't need the warning
        np.seterr(divide='ignore', invalid='ignore')

        # Divide number of foreground pixels in each row of 4 subregions by the
        # number of foreground pixels of the previous level region
        densities = n_subregion / n_region[:, None]
        
        # Handle the div by zero cases so that 0/0 = 1 for densities
        # Otherwise, we would get codeword 0000, which the paper disallows
        inds = np.where(n_region==0)[0]
        densities[inds, :] = 1

        return densities


    def calc_areas(self, subregions):
        """
        Get the area (total number of pixels) of each of the given subregions.
        """
        areas = np.array([x.size for x in subregions])
        return areas
    
    
    def count_codes(self, rel_densities):
        """
        Each row of rel_densities is treated as a codeword. The frequency of codewords
        is tabulated (with 15 possible codewords, because 0b0000 is not possible). The 
        returned code_count values are real values between 0 and 1.

        Args
            rel_densities (binary 4^levelx4 ndarray): The quantized relative densities
        
        Return
            code_count (real 15x1 ndarray): Histogram of code [0b0001 - 0b1111] frequency
        """
        
        code_dict = {(False,False,False, True): 0,
                     (False,False, True,False): 1,
                     (False,False, True, True): 2,
                     (False, True,False,False): 3,
                     (False, True,False, True): 4,
                     (False, True, True,False): 5,
                     (False, True, True, True): 6,
                     ( True,False,False,False): 7,
                     ( True,False,False, True):	8,
                     ( True,False, True,False):	9,
                     ( True,False, True, True):	10,
                     ( True, True,False,False):	11,
                     ( True, True,False, True):	12,
                     ( True, True, True,False):	13,
                     ( True, True, True, True):	14}

        codes = np.array([code_dict[tuple(x)] for x in rel_densities])
        code_count = np.zeros(len(code_dict))
        for i in range(len(code_dict)):
            code_count[i] = len(np.where(codes==i)[0])
        return code_count / len(rel_densities)



### Script for running demo ###

def get_args():
    """
    Defines commandline arguments.
    """
    parser = argparse.ArgumentParser(description='Read labels from patent figures.')
    parser.add_argument('--image_path', type=str, default='./',
                        help='All images in path will be processed [default=./]')
    parser.add_argument('--image_list', type=str, default=None,
                        help='If given, process only listed images (1st column of text file from image_path')
    parser.add_argument('--output', type=str, default='output',
                        help='Result table save to [OUTPUT_DIR]/[OUTPUT].csv [default=output]')
    parser.add_argument('--save_all_images', action='store_true', default=False,
                        help='If enabled, all intermediate processing steps will be saved as images to OUTPUT_DIR')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to store results in [default=./]')
    # Algorithm parameters
    parser.add_argument('--n_levels', type=int, default=10,
                        help='Number of hierarchy levels [default=10]')
    parser.add_argument('--n_nonquant_levels', type=int, default=2,
                        help='All levels >= n_nonquant_levels will be quantized [default=2]')

    return parser.parse_args()
    

def main():
    args = get_args()

    if args.image_list is None:
        img_fnames = glob.glob(args.image_path + '/*')
    else:
        with open(args.image_list) as f_in:
            img_fnames = [args.image_path + '/' + line.split()[0] for line in f_in]

    im_sig = AdaptiveHistogram(args.n_levels, args.n_nonquant_levels,
                               save_all_images=args.save_all_images)

    # Calculation is slow, so save each hash as it calculated
    with open(args.output_dir + '/' + args.output + '.csv', 'w') as f:
        print(time.asctime(), "Start processing ", len(img_fnames), " images")
        start = time.time()
        count = 0
        for fname in img_fnames:
            features = im_sig.generate_signature(fname)
            if features is None:
                print('No features found for ' + fname)
                continue
            features_str_array = [str(x) for x in features]
            f.write(fname + ',' + ','.join(features_str_array) + '\n')
            count += 1

    end = time.time()
    print(time.asctime(), 'Done processing', count, 'images in', end-start, 'seconds')
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
