
import numpy as np

external = np.diag([1, 1, 1, 1]).reshape(4, 2, 2)
internal = np.logical_not(external)
cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])

def match(a, masks):
    for mask in masks:
        if np.all(a == mask):
            return True
    return False

def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4

print("The number of objects in the image:")

image = np.load("./example1.npy")
image[image != 0] = 1
print("example1.npy:",count_objects(image))

image_ = np.load("./example2.npy")

for i in range(image_.shape[-1]):
    image_[image_ != 0] = 1
    
print("example2.npy:", np.sum(count_objects(image_[:, :, i]) for i in range(image_.shape[-1])))