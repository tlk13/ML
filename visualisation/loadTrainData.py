import gzip
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys


currentPathIm = os.path.dirname(os.path.abspath(__file__))
newPathIm = currentPathIm.replace("visualisation","data/trainImages.gz")

currentPathLa = os.path.dirname(os.path.abspath(__file__))
newPathLa = currentPathLa.replace("visualisation","data/trainLabels.gz")

f = gzip.open(newPathIm, 'rb')
g = gzip.open(newPathLa, 'rb')

imageSize = 28
numImages = 60000
x_train = []; y_train = []


buf = f.read(imageSize*imageSize*numImages)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(numImages, imageSize*imageSize)
data /= data.max()
x_train = data

for i in range(numImages):
    buf = g.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    if i<10: print(labels)
    y_train.append(labels)
