### ©2020. Triad National Security, LLC. All rights reserved.
### This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
###
###

from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread
from PIL import Image
try:
    from cairosvg import svg2png
except ImportError:
    pass
from io import BytesIO
import numpy as np
import xml.etree



class ImageSignature(object):
    """Image signature generator.
    """

    def __init__(self, n=9, crop_percentiles=(1, 99), patch_width=None,
                 n_levels=3, proportional_patch=True, threshold=None):
        """Initialize the signature generator.

        Args:
            n (Optional[int]): size of grid imposed on image. Grid is n x n (default 9)
            crop_percentiles (Optional[Tuple[int]]): lower and upper bounds when considering how much
                variance to keep in the image (default (1, 99))
            patch_width (Optional[int]): width in pixels of sample region. If none, the patch width is
                min(width, height) of the image divided by n (default None)
            n_levels (Optional[int]): number of positive and negative groups to stratify neighbor
                differences into. n = 3 -> [-3, -2, -1, 0, 1, 2, 3] (default 3)
            proportional_patch (Optional[boolean]): Automatically adjust patch height so that patches
                have the same aspect ratio as the original image, otherwise the patch height is equal 
                to the patch width (default True)
            threshold (Optional[float]): threshold between 0 and 1 for considering difference of 
                neighboring regions to be identical in mean gray value, or None to automatically
                calculate this threshold based on patch size (default None)
        """

        # check inputs
        assert crop_percentiles is None or len(crop_percentiles) == 2,\
            'crop_percentiles should be a two-value tuple, or None'
        if crop_percentiles is not None:
            assert crop_percentiles[0] >= 0,\
                'Lower crop_percentiles limit should be >= 0 (%r given)'\
                % crop_percentiles[0]
            assert crop_percentiles[1] <= 100,\
                'Upper crop_percentiles limit should be <= 100 (%r given)'\
                % crop_percentiles[1]
            assert crop_percentiles[0] < crop_percentiles[1],\
                'Upper crop_percentile limit should be greater than lower limit.'
            self.lower_percentile = crop_percentiles[0]
            self.upper_percentile = crop_percentiles[1]
            self.crop_percentiles = crop_percentiles
        else:
            self.crop_percentiles = crop_percentiles
            self.lower_percentile = 1
            self.upper_percentile = 99

        assert type(n) is int, 'n should be an integer > 1'
        assert n > 1, 'n should be greater than 1 (%r given)' % n
        self.n = n

        if patch_width is not None:
            assert type(patch_width) is int, 'patch_width should be an integer >= 1, or None'
            assert patch_width >= 1, 'patch_width should be greater than 0 (%r given)' % patch_width
        self.patch_width = patch_width

        assert type(proportional_patch) is bool, 'proportional_patch should be boolean (%r given)' % proportional_patch
        self.proportional_patch = proportional_patch

        assert type(n_levels) is int, 'n_levels should be an integer'
        assert n_levels > 0, 'n_levels should be > 0 (%r given)' % n_levels
        self.n_levels = n_levels

        if threshold is not None:
            assert type(threshold) is float, 'threshold should be between 0 and 1'
            assert threshold >= 0 and threshold <= 1, 'threshold should be between 0 and 1'
        self.threshold = threshold
        
        
    def generate_signature(self, path_or_image, bytestream=False):
        """Generates an image signature.

        Args:
            path_or_image (string or numpy.ndarray): image path, or image array

        Returns:
            The image signature: A numpy vector of length 2n(n-1) for an nxn grid
        """

        # Load image as array of gray-levels
        self.preprocess_image(path_or_image, bytestream=bytestream)
        
        # Determine cropping boundaries
        if self.crop_percentiles is not None:
            self.crop_image()
        else:
            self.image_limits = None
            
        # Generate grid centers
        self.compute_grid_points()

        # Compute gray level mean of each patch centered at each grid point
        self.compute_gray_means()
                                
        # Compute array of differences for each grid point and each neighbor
        self.compute_differentials()
        
        # Threshold differences that are very close
        self.threshold_diffs()

        # Bin differences to only 2q+1 values where q is n_levels
        self.bin_diffs()
        self.signature = self.level_vec
        
        return self.signature


    @staticmethod
    def convert_color(img):
        """
        From a numpy array (dtype=np.uint8), returns a 2-d numpy array of the grayscale image.
        """
        if len(img.shape) > 2:
            if img.shape[2] == 4:
                # Check if non-transparency is mostly white (then make background black)
                alpha = img[:, :, 3]
                inds = alpha > 0
                mean_pixel_value = np.mean(img[inds, :3])
                if (mean_pixel_value > 250):
                    img = rgba2rgb(img, (0,0,0))
                else:
                    img = rgba2rgb(img)
            if img.shape[2] == 3:
                img = rgb2gray(img)

        return img

    
    def preprocess_image(self, image_or_path, bytestream=False):
        """Loads an image and converts to grayscale.

        Args:
            image_or_path (string or numpy.ndarray): image path, or image array
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)
        Returns:
            Array of floats corresponding to grayscale level at each pixel
        """

        if bytestream:
            try:
                img = Image.open(BytesIO(image_or_path))
            except IOError:
                # could be an svg, attempt to convert
                try:
                    img = Image.open(BytesIO(svg2png(image_or_path)))
                except (NameError, xml.etree.ElementTree.ParseError):
                    pass
            img = np.asarray(img, dtype=np.uint8)         
            self.im_array = ImageSignature.convert_color(img)
            return self.im_array
        
        elif type(image_or_path) is str:
            # Prefer imread over Image.open, except for formats that are not supported
            # For png format, imread works better than Image.open
            if image_or_path.endswith('svg'):
                # skimage, PIL do not support svg
                try:
                    img = Image.open(BytesIO(svg2png(url=image_or_path, write_to=None)))
                except (NameError, xml.etree.ElementTree.ParseError):
                    pass
            elif image_or_path.endswith('webp'):
                # skimage does not support webp
                img = Image.open(image_or_path)
            else:
                # Prefer to use skimage
                img = imread(image_or_path)
            img = np.asarray(img, dtype=np.uint8)
            self.im_array = ImageSignature.convert_color(img)
            return self.im_array
        
        elif type(image_or_path) is bytes:
            try:
                img = imread(image_or_path)
            except:
                # try again if format not suppoted by skimage
                img = Image.open(image_or_path)
            img = np.asarray(img, dtype=np.uint8)         
            self.im_array = ImageSignature.convert_color(img)
            return self.im_array

        elif type(image_or_path) is np.ndarray:
            img = np.asarray(image_or_path, dtype=np.uint8)
            self.im_array = ImageSignature.convert_color(img)
            return self.im_array
        
        else:
            raise TypeError('Path or image required.')


    def crop_image(self):
        """Crops an image, removing featureless border regions.

        Returns:
            A pair of tuples describing the 'window' of the image to use in analysis: [(top, bottom), (left, right)]
        """
        image = self.im_array
        # row-wise differences
        rw = np.cumsum(np.sum(np.abs(np.diff(image, axis=1)), axis=1))
        # column-wise differences
        cw = np.cumsum(np.sum(np.abs(np.diff(image, axis=0)), axis=0))

        # compute percentiles
        upper_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, self.upper_percentile),
                                             side='left')
        lower_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, self.lower_percentile),
                                             side='right')
        upper_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, self.upper_percentile),
                                          side='left')
        lower_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, self.lower_percentile),
                                          side='right')

        # if image is nearly featureless, use whole image
        if lower_row_limit >= upper_row_limit:
            lower_row_limit = int(0)
            upper_row_limit = int(image.shape[0])
        if lower_column_limit >= upper_column_limit:
            lower_column_limit = int(0)
            upper_column_limit = int(image.shape[1])

        # Return limits
        self.image_limits = [(lower_row_limit, upper_row_limit),
                (lower_column_limit, upper_column_limit)]
        return self.image_limits

    

    def compute_grid_points(self):
        """Computes grid centers by evenly spacing across cropped image.

        Returns:
            tuple of arrays indicating the vertical and horizontal locations of the grid points

        """
        window = self.image_limits
        
        # spread patches out evenly across image, so that they can be tiled
        # Need to account for cropping when calculating patch width
        image_width = window[0][1] - window[0][0]
        image_height = window[1][1] - window[1][0]

        # Calculate the upper limit of the patch size given n and dimensions of cropped image
        if self.proportional_patch:
            width_max = image_width / self.n
            height_max = image_height / self.n
        else: # square patch
            width_max = min(image_width, image_height) / self.n
            height_max = width_max

        # Grid sections are centered on maximum patches
        self.x_coords = np.linspace(window[0][0] + width_max/2, window[0][1] - width_max/2, self.n, dtype=int)
        self.y_coords = np.linspace(window[1][0] + height_max/2, window[1][1] - height_max/2, self.n, dtype=int)

        # Keep track of maximum patch size
        self.width_max = int(width_max)
        self.height_max = int(height_max)

        # Return coordinate pairs of grid centers
        return self.x_coords, self.y_coords

    
    def compute_gray_means(self):
        """Computes array of grayness means.

        Returns:
            an n x n array of average grayscale around the gridpoint, where n is the
                number of grid points in each direction
        """
        image = self.im_array
        num_x = self.x_coords.shape[0]
        num_y = self.y_coords.shape[0]
        window = self.image_limits
        
        # First, create the patches
        if self.patch_width is None:
            if self.proportional_patch:
                # Size of patch fills each direction
                self.patch_width = self.width_max # calculated when assigning grid points
                self.patch_height = self.height_max
            else:
                # Patch will fill one direction but not overflow the other, the patch is square
                self.patch_width = np.min([self.width_max, self.height_max])
                self.patch_height = self.patch_width
        else:
            # Note, there is currently no checking whether patch overflows the image or
            # causes the patches to overlap
            if self.proportional_patch:
                # Adjust height relative to dimensions of image
                image_width = window[0][1] - window[0][0]
                image_height = window[1][1] - window[1][0]
                ratio = image_height // image_width
                self.patch_height = int(ratio * self.patch_width)
            else:
                # Patch will be square
                self.patch_height = self.patch_width
        # Check that we have a valid patch size
        assert self.patch_width > 0, 'Width of patch must be at least 1 pixel (%r calculated)' %self.patch_width
        assert self.patch_height > 0, 'Height of patch must be at least 1 pixel (%r calculated)' %self.patch_height

        
        # Second, calculate mean gray value per patch
        avg_gray = np.zeros((num_x, num_y))
        half_width = self.patch_width/2 # only need to calculate this once
        half_height = self.patch_height/2
        
        for i, x in enumerate(self.x_coords):
            lower_x_lim = int(max([x - half_width, 0]))
            upper_x_lim = int(min([lower_x_lim + self.patch_width, image.shape[0]]))
            for j, y in enumerate(self.y_coords):
                lower_y_lim = int(max([y - half_height, 0]))
                upper_y_lim = int(min([lower_y_lim + self.patch_height, image.shape[1]]))

                avg_gray[i, j] = np.mean(image[lower_x_lim:upper_x_lim,
                                        lower_y_lim:upper_y_lim])

        self.gray_level_matrix = avg_gray
        return self.gray_level_matrix


    def compute_differentials(self):
        """Computes differences in graylevels for neighboring patches.

        The values are the differences between neighboring patches, in this order:
        left:        n*(n-1) values
        up:          n*(n-1) values
        
        Within each portion of the list, the grid patches are such:
        left:        [(0,1), (0, 2), ..., (0,n), (1,1), (1,2), ..., (1,n), (2,1), ...]
                     # column 0 has no left neighbors
        up:          [(1,0), (1,1), ..., (1,n), (2,0), (2,1), ..., (2,n), (3,1), ...]
                     # row 0 has no up neighbors

        Returns:
            a numpy vector of length 2n(n-1) for an n x n grid
        """

        # returns n x (n-1) matrix
        left_neighbors = np.diff(self.gray_level_matrix)
        # returns (n-1) x n matrix
        up_neighbors = np.diff(self.gray_level_matrix, axis=0)

        # axis=None flattens mats row-major before concatenating
        self.diff_vec = np.concatenate((left_neighbors, up_neighbors), axis=None)

        return self.diff_vec

    
    def threshold_diffs(self):
        """
        Difference vector is thresholded so that near-zero values are set to zero.
        Fixed threshold can be specified, or if none, calculate threshold based on
        size of patch
        """

        # set very close values as equivalent
        if self.threshold is None:
            # Identical tolerance is 1 / (sqrt(6 * w * h) where w and h are the patch_width and patch_height
            self.threshold = 1. / (8*np.sqrt(6) * np.sqrt(self.patch_width) * np.sqrt(self.patch_height))
        
        mask = np.abs(self.diff_vec) < self.threshold
        self.diff_vec[mask] = 0.

        return self.diff_vec
    
    
    def bin_diffs(self, wbg_version=False):
        """ Defines ordinal levels from difference vector.
        Creates level_vec: a vector of same length as diff_vec, 
        but with integer values representing the levels of differences.
        The output of this is the image signature.
        """
        
        # Create a new vector for levels of difference
        self.level_vec = np.zeros(self.diff_vec.shape)

        # if image is essentially featureless, exit here; returning all-zeros signature
        if np.all(self.diff_vec == 0):
            return self.level_vec

        # WBG way:
        # bin so that size of bins on one side of zero are equally popular
        if wbg_version:
            positive_cutoffs = np.percentile(self.diff_vec[self.diff_vec > 0.],
                                             np.linspace(0, 100, self.n_levels+1))
            negative_cutoffs = np.percentile(self.diff_vec[self.diff_vec < 0.],
                                             np.linspace(100, 0, self.n_levels+1))

            # Last value in percentile cutoffs is meaningless
            for cutoff in positive_cutoffs[:-1]:
                self.level_vec[self.diff_vec >= cutoff] += 1

            # Last value in percentile cutoffs is meaningless
            for cutoff in negative_cutoffs[:-1]:
                self.level_vec[self.diff_vec <= cutoff] -= 1
            return self.level_vec
        

        # Bin so that size of bins are equally popular for absolute values
        # This preserves the bin boundaries for comparing flipped images where the differences will
        # have opposite sign.
        abs_cutoffs = np.percentile(np.abs(self.diff_vec), np.linspace(0, 100, self.n_levels+1))

        # Last value in percentile cutoffs is meaningless
        for cutoff in abs_cutoffs[:-1]:
            self.level_vec[self.diff_vec >= cutoff] += 1
            self.level_vec[self.diff_vec <= -1*cutoff] -= 1

        
        return self.level_vec
