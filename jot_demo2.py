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

# Plot and show
plt.figure(1)
plt.imshow(y_raw, cmap="gray")
plt.title("raw y[0]")
plt.figure(2)
plt.imshow(y[0], cmap="gray")
plt.title("p(k) > {} for y[0]".format(q))
plt.figure(3)
plt.imshow(y_mean, cmap="gray")
plt.title("Simple Sum for {}".format(alpha))
plt.show()

