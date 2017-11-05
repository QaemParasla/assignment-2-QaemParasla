# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import cv2
import numpy as np
import math
import cmath

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""


        #Compared it to numpy fft and it gave exactly same results as the algo below
        row, col = matrix.shape
        dft = np.zeros((row,col), dtype=np.complex_)

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i,j]*(math.cos((2*math.pi/15)*((u*i)+(v*j))) - cmath.sqrt(-1) * math.sin((2*math.pi/15)*((u*i)+(v*j))))
                dft[u,v] = sum
                sum = 0

        return dft

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        #given matrix is DFT
        """
        npfshift = np.fft.fftshift(matrix)
        im = Image.fromarray(npfshift.real)
        im.show()
        print("NUMPY START------------------------------------------------------------------------NUMPY START ")
        print(npfshift)
        print("NUMPY END------------------------------------------------------------------------NUMPY END ")
        """

        row, col = matrix.shape
        idft = np.zeros((row,col), dtype=np.complex_)

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i,j]*(math.cos((2*math.pi/15)*((u*i)+(v*j))) + cmath.sqrt(-1) * math.sin((2*math.pi/15)*((u*i)+(v*j))))
                idft[u,v] = sum
                sum = 0

        """
        im = Image.fromarray(idft.real)
        im.show()
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
        im = Image.fromarray(compDCT.real)
        im.show()
        print("COMPUTER DCT START-----------------------------------------------------------------------COMPUTER START")
        print(compDCT)
        print("COMPUTER DCT END-----------------------------------------------------------------------COMPUTER END")
        """

        row, col = matrix.shape
        dct = np.zeros((row, col))

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i, j] * (math.cos((2 * math.pi / 15) * ((u * i) + (v * j))))
                dct[u, v] = sum
                sum = 0

        """
        im = Image.fromarray(dct.real)
        im.show()
        print("MY DCT START-------------------------------------------------------------------  MY DCT START")
        print(dct)
        print("MY DCT END-----------------------------------------------------------------------MY DCT END")
        """

        return dct


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        #given matrix is already in idft
        row, col = matrix.shape
        mag_matrix = np.zeros((row, col))

        for u in range(row):
            for v in range(col):
                mag_matrix[u,v] = math.sqrt(matrix[u,v].real ** 2 + matrix[u,v].imag ** 2)

        return mag_matrix
