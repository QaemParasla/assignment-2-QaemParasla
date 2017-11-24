# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import cv2
import numpy as np
from scipy import fftpack
from PIL import Image
import time

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        #Compared it to numpy fft and it gave exactly same results as the algo below
        """
        npfft2 = np.fft.fft2(matrix)
        #im = Image.fromarray(npfft2.real)
        #im.show()
        print("NUMPY FFT2 START------------------------------------------------------------------------NUMPY FFT2 START ")
        print(npfft2)
        print("NUMPY FFT2 END  ------------------------------------------------------------------------NUMPY FFT2 END   ")
        """

        row, col = matrix.shape
        myDFT = np.zeros((row,col), dtype=np.complex_)

        #start = time.time()

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i,j]*(np.cos((2*np.pi/row)*((u*i)+(v*j))) - 1j * np.sin((2*np.pi/row)*((u*i)+(v*j))))
                        #sum += matrix[i,j] * cmath.exp(- 1j * 2*math.pi * (float(u * i) / row + float(v * j) / col))
                myDFT[u,v] = sum
                sum = 0

        #stop  = time.time()
        #print(stop-start)

        return myDFT


    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        """
        npifft2 = np.fft.ifft2(matrix, s=None, axes=None, norm=None)
        #im = Image.fromarray(npifft2.real)
        #im.show()
        print("NUMPY IFFT2 START------------------------------------------------------------------------NUMPY IFFT2 START ")
        print(npifft2)
        print("NUMPY IFFT2 END  ------------------------------------------------------------------------NUMPY IFFT2 END   ")
        """

        row, col = matrix.shape
        idft = np.zeros((row,col), dtype=np.complex_)

        sum = 0
        for i in range(row):
            for j in range(col):
                for u in range(row):
                    for v in range(col):
                        sum += matrix[u,v]*(np.cos((2*np.pi/row)*((u*i)+(v*j))) + 1j * np.sin((2*np.pi/row)*((u*i)+(v*j))))
                        #sum += matrix[u, v] * cmath.exp(1j * (2 * math.pi / 3) * ((u * i) + (v * j)))
                idft[i,j] = sum / (row * col)
                sum = 0

        """
        #im = Image.fromarray(idft.real)
        #im.show()
        print("MY IDFT START-----------------------------------------------------------------------MYIDFT START")
        print(idft)
        print("MY IDFT END-----------------------------------------------------------------------MYIDFT END")
        """

        return idft


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        """
        compDCT = fftpack.dct(fftpack.dct(matrix.T, norm=None).T, norm=None)
        #compDCT = fftpack.dct(matrix, type=None, n=matrix.shape, axis=None, norm=None, overwrite_x=None)
        im = Image.fromarray(compDCT.real)
        im.show()
        print("NUMPY DCT START-----------------------------------------------------------------------NUMPY DCT START")
        print(compDCT)
        print("NUMPY DCT END  -----------------------------------------------------------------------NUMPY DCT END  ")
        """


        def alpha(a):
            if a == 0:
                return np.sqrt(1.0 / 8)
            else:
                return np.sqrt(2.0 / 8)

        row, col = matrix.shape
        myDCT = np.zeros((row, col))

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += alpha(u) * alpha(v) * np.cos(((2 * i + 1) * (u * np.pi)) / (2 * 8)) * np.cos(((2 * j + 1) * (v * np.pi)) / (2 * 8))
                myDCT[u, v] = sum
                sum = 0

        """
        im = Image.fromarray(dct.real)
        im.show()
        print("MY DCT START-------------------------------------------------------------------  MY DCT START")
        print(dct)
        print("MY DCT END-----------------------------------------------------------------------MY DCT END")
        """

        return myDCT

    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        #given matrix is already in idft so have to convert it to dft and then take the magnitude

        myDFT = self.forward_transform(matrix)

        row, col = myDFT.shape
        mag_matrix = np.zeros((row, col))

        for u in range(row):
            for v in range(col):
                mag_matrix[u,v] = np.sqrt(myDFT[u,v].real ** 2 + myDFT[u,v].imag ** 2)

        return mag_matrix
