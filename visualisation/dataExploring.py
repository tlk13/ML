# This is a python file to explore the data as received in the MNIST database
import gzip
import os.path
import numpy as np
import matplotlib.pyplot as plt

currentPath = os.path.dirname(os.path.abspath(__file__))
print(currentPath)
newPath = currentPath.replace("visualisation","data/trainImages.gz")



f = gzip.open(newPath, 'r')

imageSize = 28
numImages = 5


buf = f.read(imageSize*imageSize*numImages)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(numImages, imageSize, imageSize, 1)
data /= data.max()


for x in data:
    for y in x:
        for z in range(imageSize):
            print(y[z], end="")
        print("")
    print("\n" + 80*"#" + "\n")
