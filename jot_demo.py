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
raw_jot_data = np.empty(shape=(12,3024,3024))
data_cube = np.empty(shape=(12,504,504))

# Fetch raw image data
print("Loading raw data...")
for i in range(0, 12):
    name = "images/IMG_{}.jpg".format(i+1)
    raw_jot_data[i] = cv2.imread(name, 0)/255

# Downsample raw data to shape (12, 504, 504)
raw_jot_data = raw_jot_data[::1,::6,::6]

# Define function to determine bit state
def P(k, lim):
    if (k > lim):
        return 1
    else:
        return 0

# Apply P(k) to raw data to generate data cube
print("Generating data cube...")
p_vec = np.vectorize(P)
data_cube_1 = p_vec(raw_jot_data, 0.2)
data_cube_2 = p_vec(raw_jot_data, 0.25)
data_cube_3 = p_vec(raw_jot_data, 0.3)
data_cube_4 = p_vec(raw_jot_data, 0.4)
data_cube_5 = p_vec(raw_jot_data, 0.5)
data_cube_6 = p_vec(raw_jot_data, 0.75)

# Average out the bits in the temporal axis
print("Averaging values on temporal axis...")
output_1 = np.apply_along_axis(np.mean, 0, data_cube_1)
output_2 = np.apply_along_axis(np.mean, 0, data_cube_2)
output_3 = np.apply_along_axis(np.mean, 0, data_cube_3)
output_4 = np.apply_along_axis(np.mean, 0, data_cube_4)
output_5 = np.apply_along_axis(np.mean, 0, data_cube_5)
output_6 = np.apply_along_axis(np.mean, 0, data_cube_6)

# Plot and show
print("Plotting output...")
plt.figure(1)
plt.imshow(output_1, cmap="gray")
plt.title("p(k) > 0.2")
plt.figure(2)
plt.imshow(output_2, cmap="gray")
plt.title("p(k) > 0.25")
plt.figure(3)
plt.imshow(output_3, cmap="gray")
plt.title("p(k) > 0.3")
plt.figure(4)
plt.imshow(output_4, cmap="gray")
plt.title("p(k) > 0.4")
plt.figure(5)
plt.imshow(output_5, cmap="gray")
plt.title("p(k) > 0.5")
plt.figure(6)
plt.imshow(output_6, cmap="gray")
plt.title("p(k) > 0.75")
plt.figure(7)
plt.imshow(raw_jot_data[1], cmap="gray")
plt.title("original")
plt.show()

