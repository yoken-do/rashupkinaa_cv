import numpy as np
from skimage.measure import label
from skimage.morphology import binary_opening

image = np.load("./stars.npy")

cross = np.eye(5)[::-1] + np.eye(5)
cross[cross > 0] = 1

plus = np.zeros((5, 5))
plus[:, 2] = 1
plus[2, :] = 1

cross_, plus_ = binary_opening(image, cross), binary_opening(image, plus)

count = np.max(label(cross_)) + np.max(label(plus_))

print("Stars in the image:", count)