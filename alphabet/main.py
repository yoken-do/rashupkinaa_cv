import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from pathlib import Path

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def count_vlines(region):
    return np.all(region.image, axis = 0).sum()

def count_lgr_vlines(region):
    x = region.image.mean(axis = 0) == 1
    return np.sum(x[:len(x)//2]) > np.sum(x[len(x)//2:])

def recognize(region):
    if np.all(region.image):
        return "-"
    else:
        holes = count_holes(region)
        if holes == 2:
            vlines = count_vlines(region)
            if count_lgr_vlines(region) and vlines > 3:
                return "B"
            else:
                return "8"
        elif holes == 1:
            inv_image = ~region.image
            labeled = label(inv_image)
            if region.eccentricity < 0.6 and np.max(labeled) == 3:
                return "D"
            elif region.eccentricity > 0.6 and np.max(labeled) == 3:
                return "P"
            elif np.max(labeled) == 4:
                return "A"
            elif np.max(labeled) == 5:
                return "0"
        else:
            if count_vlines(region) >= 3:
                return "1"
            else:
                if region.eccentricity <= 0.5:
                    return "*"
                inv_image = ~region.image
                inv_image = binary_dilation(inv_image, np.ones((3, 3)))
                labeled = label(inv_image, connectivity=1)
                match np.max(labeled):
                    case 2: return "/"
                    case 4: return "X"
                    case _: return "W"

    return "#"

symbols = plt.imread(Path(__file__).parent / "symbols.png")[:,:,:-1]
gray = symbols.mean(axis = 2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

result = {}

out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)

plt.figure()
for i, region in enumerate(regions):
    symbol = recognize(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(symbol)
    plt.imshow(region.image)
    plt.savefig(out_path / f"{i:03d}.png")

print(result)