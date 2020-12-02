import itertools as it
import numpy as np





grids_center_coordi = it.product(np.arange(
    0, 5), np.arange(0, 5))
grids_center_coordi = np.array([x for x in grids_center_coordi])
print(grids_center_coordi)