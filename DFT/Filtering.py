# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import cv2
import numpy as np
import math
import cmath

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order=0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order

    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""

        row, col = shape
        mask = np.zeros([row, col])

        for u in range(row):
            for v in range(col):
                if np.sqrt((u - row / 2) ** 2 + (v - col / 2) ** 2) <= cutoff:
                    mask[u,v] = 1

        return mask

    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        # Hint: May be one can use the low pass filter function to get a high pass mask

        row, col = shape
        mask = np.zeros([row, col])

        for u in range(row):
            for v in range(col):
                if np.sqrt((u - row / 2) ** 2 + (v - col / 2) ** 2) > cutoff:
                    mask[u, v] = 1

        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        row, col = shape
        mask = np.zeros([row, col])

        for u in range(row):
            for v in range(col):
                mask[u,v] = 1 / (1 + (np.sqrt((u-row/2)**2+(v-col/2)**2) / cutoff) ** (2 * order))

        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        # Hint: May be one can use the low pass filter function to get a high pass mask
        row, col = shape
        mask = np.zeros([row, col])

        for u in range(row):
            for v in range(col):
                mask[u,v] = 1 / (1 + cutoff / (np.sqrt((u - row / 2) ** 2 + (v - col / 2) ** 2)) ** (2 * order))

        return mask


    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""

        row, col = shape
        mask = np.zeros([row, col])

        for u in range(row):
            for v in range(col):
                mask[u,v] = np.exp(((-np.sqrt((u - row / 2) ** 2 + (v - col / 2) ** 2)) ** 2) / (2 * (cutoff ** 2)))

        return mask


    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        # Hint: May be one can use the low pass filter function to get a high pass mask

        row, col = shape
        mask = np.zeros([row, col])

        for u in range(row):
            for v in range(col):
                mask[u, v] = 1-np.exp(((-np.sqrt((u - row / 2) ** 2 + (v - col / 2) ** 2)) ** 2) / (2 * (cutoff ** 2)))

        # im = Image.fromarray(mask.real)
        # im.show()

        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """

        #1
        row, col = image.shape
        fcs = np.zeros((row, col), dtype=np.uint8)
    
        max = image.real.max()
        min = image.real.min()
        maxminusmin = max - min
        k = 254

        for x in range(row):
            for y in range(col):
                fcs[x,y] = math.floor(((k/maxminusmin)*image[x,y]-1) + 0.5)

        return fcs

    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """

        #np.set_printoptions(threshold=np.nan)

        # 1 compute the fft of the image
        dft = np.fft.fft2(self.image, s=None, axes=None, norm=None)
        row, col = dft.shape
        mag_matrix = np.zeros((row, col))

        for u in range(row):
            for v in range(col):
                mag_matrix[u, v] = math.sqrt(dft[u, v].real ** 2 + dft[u, v].imag ** 2)
        dftimg = self.post_process_image(mag_matrix)

        #2. shift the fft to center the low frequencies
        fshift = np.fft.fftshift(dft, axes=None)

        #3. get the mask
        if self.filter.__name__ == "get_ideal_low_pass_filter" or "get_ideal_high_pass_filter":
            mask = self.filter(self.image.shape, self.cutoff)
        else:
            mask = self.filter(self.image.shape, self.cutoff, self.order)

        #4 Convolution theorem)


        #5 compute the inverse shift
        inverseShift = np.fft.ifftshift(fshift)

        #6 compute the inverse fourier transform
        ift = np.fft.ifft(inverseShift, n=None, axis=-1, norm=None)

        fcs = self.post_process_image(ift)

        #7 magnitude
        row, col = ift.shape
        mag_matrix = np.zeros((row, col))

        for u in range(row):
            for v in range(col):
                mag_matrix[u, v] = math.sqrt(ift[u, v].real ** 2 + ift[u, v].imag ** 2)


        return [self.image, dftimg, self.image]
