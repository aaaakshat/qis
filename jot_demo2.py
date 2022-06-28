#!/usr/bin/env python3

# Akshat Bisht, 2021.
# arb@purdue.edu

import numpy as np
import matplotlib.pyplot as plt
import cv2
from bm3d import bm3d
from ttictoc import tic, toc

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

print("bottleneck")

p = 1-(np.e**x)

print(f"range of p max: {np.max(p)} x min: {np.min(p)}")
print(np.isnan(p).any())
print(np.max(p))
print(np.any(p < 0))
print(p)

#tic()
#y_af = np.random.binomial(10, p)
#y_af = np.where(y_af<q, 0, 1)
#print("Yaf: ", y_af.shape)
#print(toc())


tic()
y = np.random.binomial(T,1-np.exp(-alpha*x))
xhat  = bm3d(y/100, 0.04)*100
plt.imshow(xhat,cmap='gray')
plt.show()
print("bm3d", toc())


x_T = np.repeat(x[:, :, np.newaxis], 10, axis=2)
print('x_t shape: ', x_T.shape)

y_new = np.random.poisson(alpha*x_T)
y_new = np.where(y_new<q, 0, 1)

plt.show()

plt.figure(1)
plt.imshow(y_new[:,:,1], cmap='gray')
y_new = np.apply_along_axis(np.sum, 0, y)
print("ynew dim", y_new.shape)


tic()
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

y_mean = np.apply_along_axis(np.sum, 0, y)

print(toc())
plt.show()

# Plot and show
plt.figure(1)
plt.imshow(y_raw, cmap="gray")
plt.title("raw y[0]")
plt.figure(2)
plt.imshow(y[0], cmap="gray")
plt.title("p(k) > {} for y[0]".format(q))
plt.figure(3)
plt.imshow(y_mean, cmap="gray")
print("dim: ", y_mean.shape)
plt.title("Simple Sum for {}".format(alpha))
plt.show()

