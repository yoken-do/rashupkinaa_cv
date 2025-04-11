
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path

image = plt.imread("./alphabet-small.png")

def extractor(region):
    area = region.area / region.image.size
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter
    perimeter /= region.image.size
    eccentricity = region.eccentricity
    vlines = np.sum(region.image, 0) == region.image.shape[0]
    vlines = np.sum(vlines) / region.image.shape[1]
    x = region.image.mean(axis=0) == 1
    holes = count_holes(region)
    ratio = region.image.shape[1] / region.image.shape[0]
    return np.array([area, cy, cx, perimeter, eccentricity, vlines, holes, abs(cx - cy), ratio, cx < 0.44])

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def norm_l1(v, v_):
    return ((v - v_) ** 2).sum() ** 0.5

def classificator(v, template):
    result = "_"
    min_dist = 10 ** 16
    for key in templates:
        d = norm_l1(v, templates[key])
        if d < min_dist:
            result = key
            min_dist = d
    return result
  
gray = image.mean(axis=2)
b = gray < 1
labeled = label(b)
regions = regionprops(labeled)
print(len(regions))

templates = {"A" : extractor(regions[2]), 
             "B" : extractor(regions[3]), 
             "8" : extractor(regions[0]), 
             "0" : extractor(regions[1]), 
             "1" : extractor(regions[4]), 
             "W" : extractor(regions[5]), 
             "X" : extractor(regions[6]), 
             "*" : extractor(regions[7]), 
             "-" : extractor(regions[9]), 
             "/" : extractor(regions[8])}

print(templates)

# for region in regions:
#     v = extractor(region)
#     print(classificator(v, templates))

# c = 1
# for i, region in enumerate(regions):
#     v = extractor(region)
#     plt.subplot(2, 5, c)
#     plt.title(classificator(v, templates))
#     c+=1;
#     plt.imshow(region.image)

symbols = plt.imread("./alphabet.png")[:, :, :-1]
gray = symbols.mean(axis=2)
b = gray > 0
labeled = label(b)
regions = regionprops(labeled)

out_path = Path(__file__) / "out"
out_path.mkdir(exist_ok=True)

for i, region in enumerate(regions):
    plt.cla()
    plt.figure()
    v = extractor(region)
    plt.title(classificator(v, templates))
    plt.imshow(region.image)
    plt.savefig(out_path / f"{i:03d}.png")
    
# plt.imshow(regions[1].image)
# plt.show()

# c = 1
# for symbol, region in zip(templates, regions):
#     plt.subplot(2, 5, c)
#     c+=1
#     plt.title(symbol)
#     plt.imshow(region.image)
    
# plt.imshow(labeled)
# plt.show()