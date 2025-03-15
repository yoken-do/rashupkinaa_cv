import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (binary_closing, binary_dilation, binary_erosion, binary_opening)



for j in range(1, 7, 1):
    image = np.load(f"./wires/wires{j}npy.txt")
    break_wires = 0
    result = binary_erosion(image, np.ones(3).reshape(3, 1))
    labeled, res = label(image), label(result)
    num_wires = np.max(labeled)

    for i in range(1, num_wires + 1, 1):
        mask = (labeled == i)

        before, after = np.max(label(mask * image)), np.max(label(mask * res))
        if before >= after: 
            print(f"The wire {i} in the image {j} has no gap")
        else: 
            print(f"The wire {i} in the image {j} has a gap {after} the blocks")
    