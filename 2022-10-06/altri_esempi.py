import numpy as np

arr = np.arange(10)
print(arr)
slice = arr[2:5]
slice[:] = 12
print(arr)
slice = slice / 2
print(arr)

