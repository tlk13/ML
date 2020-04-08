import gzip
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys


currentPathIm = os.path.dirname(os.path.abspath(__file__))
newPathIm = currentPathIm.replace("visualisation","data/trainImages.gz")

currentPathLa = os.path.dirname(os.path.abspath(__file__))
newPathLa = currentPathLa.replace("visualisation","data/trainLabels.gz")

f = gzip.open(newPathIm, 'r')
g = gzip.open(newPathLa, 'r')

imageSize = 28
numImages = 60000



buf = f.read(imageSize*imageSize*numImages)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(numImages, imageSize, imageSize, 1)
data /= data.max()

for i in range(numImages):
    buf = g.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)

print(str(sys.getsizeof(data)))
