#!/usr/bin/env python3

# Akshat Bisht, 2021.
# arb@purdue.edu

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Setup initial cubicles
#
# Here I am reading in 12 closely timed snapshots of a greyscale scene.
# This is my proxy for the hypothetical data that would come from a QIS sensor.
# I will be running a simple P(k) function to get a binary data cube.
# Fetch raw image data
T = int(input("Enter # of frames T\n"))
q = int(input("Enter cutoff q\n"))
alpha = int(input("Enter alpha (gain)\n"))
d_a = int(input("Enter bilateral filter d:\n"))
d_b = int(input("Enter bilateral filter sigmaColor:\n"))
d_c = int(input("Enter bilateral filter sigmaSpace:\n"))
x = np.empty([3024,3024])
x = cv2.imread("images/IMG_1.jpg", 0)/255

# Downsample raw data to shape (504, 504)
x = x[::6,::6]

y = np.empty([T, 504,504])
for i in range(0, T):
    y[i] = np.random.poisson(lam=alpha*x, size=(504,504))

y_raw = y[0]

# Define function to determine bit state
def B(k, cutoff):
    if (k >= cutoff):
        return 1
    else:
        return 0

# Apply B(k) to raw data to generate data cube
b_vec = np.vectorize(B)

for i in range (0, T):
    y[i] = b_vec(y[i], q)

y_mean = np.apply_along_axis(np.mean, 0, y)

def anscombe(x):
    '''
    Compute the anscombe variance stabilizing transform.
      the input   x   is noisy Poisson-distributed data
      the output  fx  has variance approximately equal to 1.
    Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
    binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
    '''
    return 2.0*np.sqrt(x + 3.0/8.0)

def inverse_anscombe(z):
    '''
    Compute the inverse transform using an approximation of the exact
    unbiased inverse.
    Reference: Makitalo, M., & Foi, A. (2011). A closed-form
    approximation of the exact unbiased inverse of the Anscombe
    variance-stabilizing transformation. Image Processing.
    '''
    return (z/2.0)**2 - 3.0/8.0

# Normalise values between 0 - 255
y_mean = np.interp(y_mean, (y_mean.min(), y_mean.max()), (0, 255))
y_mean = np.uint8(y_mean)

y_mean_denoised = cv2.bilateralFilter(y_mean, d_a, d_b, d_c)

t_before = anscombe(y_mean).astype(np.uint8)
denoised = cv2.bilateralFilter(t_before, d_a, d_b, d_c)
t_after = inverse_anscombe(denoised)

# Plot and show
plt.figure(1)
plt.imshow(y_mean, cmap="gray")
plt.title("Simple sum")
plt.figure(2)
plt.imshow(x, cmap="gray")
plt.title("Ground truth")
plt.figure(3)
plt.imshow(t_after, cmap="gray")
plt.title("denoised using anscombe transform")
plt.figure(4)
plt.imshow(y_mean_denoised, cmap="gray")
plt.title("denoised without anscombe transform")
plt.show()

