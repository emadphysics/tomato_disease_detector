import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import os
cwd = os.getcwd()

classes = os.listdir("C:\\Users\\Gebruiker\\Desktop\\CNN based Architectures\\plant\\PlantVillage\\diseases")
enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3],[4], [5], [6], [7], [8], [9]])
def names(number):
    return classes[number]
class processing:
    def __init__(self,path,dim,classes):
        self.trainPath=path
        self.dim=dim
        self.classes=classes

    def split(self):
        def display():
            s1 = []
            for i in range(self.classes):
                s1.append([i])
            return s1
        OneHotEncoder().fit(display())
        trainData = []
        trainLabel = []
        index = 0
        for dir in os.listdir(self.trainPath):
            filePaths = []
            subDir = os.path.join(self.trainPath, dir)
            for file in os.listdir(subDir):
                imgFullPath = os.path.join(subDir, file)
                filePaths.append(imgFullPath)
                img = Image.open(imgFullPath)
                x=img
                x=np.array(img.resize(self.dim))
                trainData.append(np.array(x))
                trainLabel.append(enc.transform([[index]]).toarray())
        index += 1
        trainData = np.array(trainData)
        trainLabel = np.array(trainLabel).reshape(trainData.shape[0],self.classes)
        return trainData,trainLabel