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

        row, col = matrix.shape
        dt = np.zeros((row,col), dtype=np.complex_)

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i,j]*(math.cos((2*math.pi/15)*((u*i)+(v*j))) - cmath.sqrt(-1) * math.sin((2*math.pi/15)*((u*i)+(v*j))))
                dt[u,v] = sum
                sum = 0

        return dt

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        row, col = matrix.shape
        dt = np.zeros((row,col), dtype=np.complex_)

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i,j]*(math.cos((2*math.pi/15)*((u*i)+(v*j))) + cmath.sqrt(-1) * math.sin((2*math.pi/15)*((u*i)+(v*j))))
                dt[u,v] = sum
                sum = 0

        return dt


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        row, col = matrix.shape
        dt = np.zeros((row, col))

        sum = 0
        for u in range(row):
            for v in range(col):
                for i in range(row):
                    for j in range(col):
                        sum += matrix[i, j] * (math.cos((2 * math.pi / 15) * ((u * i) + (v * j))))
                dt[u, v] = sum
                sum = 0

        return dt


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        row, col = matrix.shape
        dt = np.zeros((row, col))
        abs(matrix)
        for u in range(row):
            for v in range(col):
                dt[u,v] = math.sqrt(matrix[u,v].real * 2 + matrix[u,v].imag * 2)

        return dt
