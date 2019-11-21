from keras.applications.inception_v3 import InceptionV3
from keras import layers
import os.path as os
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

baseDir = "./datasets/flowers_split/"
convBase = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(150, 150, 3))

dataGen = ImageDataGenerator(rescale=1. / 255)
batchSize = 32


def extractFeatures(directory, sampleCount):
    features = np.zeros(shape=(sampleCount, 3, 3, 2048))
    labels = np.zeros(shape=(sampleCount))
    generator = dataGen.flow_from_directory(
        directory=directory,
        target_size=(150, 150),
        class_mode="categorical"
    )
    i = 0
    for inputsBatch, labelsBatch in generator:
        featuresBatch = convBase.predict(inputsBatch)
        features[i * batchSize: (i + 1) * batchSize] = featuresBatch
        labels[i * batchSize: (i + 1) * batchSize] = labelsBatch
        i += 1
        if i * batchSize >= sampleCount:
            break
    return features, labels

trainFeatures, trainLabels = extractFeatures(os.join(baseDir, "train/"), 2592)
validationFeatures, validationLabels = extractFeatures(os.joinI(baseDir, "validation/"), 865)
testFeatures, testLabels = extractFeatures(os.joinI(baseDir, "test/"), 866)

