import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

transformList = []
transformList.append(transforms.Resize(224))
transformList.append(transforms.ToTensor())
transformSequence_alex = transforms.Compose(transformList)

class DatasetGenerator_VAE_Ensemble(Dataset):
    def __init__(self, path, textFile):
        self.listImageLabels = []
        self.listImageUnitLabel = []
        dataPlace = path[0]
        with open(dataPlace, 'rb') as f:
            self.data = pickle.load(f)
        f.close()

        dim = len(self.data[0]['mean']) #assuming all of them have same dim
        length = len(self.data)
        self.meanData = np.ndarray(dim)
        self.logsigma = np.ndarray(dim)

        for i in range(length):
            self.meanData = np.vstack((self.meanData, self.data[i]['mean']))
            self.logsigma = np.vstack((self.logsigma, self.data[i]['logsigma']))

        self.meanData = self.meanData[1:]
        self.logsigma = self.logsigma[1:]

        pathDatasetFile = textFile[0]
        fileDescriptor = open(pathDatasetFile, "r")
        line = True

        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                imageLabel = lineItems[1:]
                imageLabel = [int(float(i)) for i in imageLabel]

                self.listImageLabels.append(imageLabel)
                self.listImageUnitLabel.append(1)

        fileDescriptor.close()

        if len(path) == 2:
            pathDatasetFile_U = textFile[1]
            fileDescriptor = open(pathDatasetFile_U, "r")
            line = True

            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imageLabel = lineItems[1:]
                    imageLabel = [int(float(i)) for i in imageLabel]

                    self.listImageLabels.append(imageLabel)
                    self.listImageUnitLabel.append(-1)

            fileDescriptor.close()

            dataPlace = path[1]
            with open(dataPlace, 'rb') as f:
                self.data = pickle.load(f)
            f.close()

            length = len(self.data)
            for i in range(length):
                self.meanData = np.vstack((self.meanData, self.data[i]['mean']))
                self.logsigma = np.vstack((self.logsigma, self.data[i]['logsigma']))

            print("ok")


    def __getitem__(self, index):
        ladderData_mu = self.meanData[index]
        ladderData_logsigma = self.logsigma[index]
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        imageLabel_unit = torch.FloatTensor(torch.from_numpy(np.array([self.listImageUnitLabel[index]])).float())
        return ladderData_mu, ladderData_logsigma, imageLabel, imageLabel_unit

    def __len__(self):
        return len(self.meanData)

class DatasetGenerator(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):

        self.listImagePaths = []
        self.listImageLabels = []
        self.listPatientIds = []
        self.transform = transform

        fileDescriptor = open(pathDatasetFile, "r")
        line = True

        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()

                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(float(i)) for i in imageLabel]


                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageData_Alex= Image.open(imagePath).convert('RGB')
        imageData_Alex = transformSequence_alex(imageData_Alex)
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform != None: imageData = self.transform(imageData)

        return imageData, imageData_Alex, imageLabel

    def __len__(self):
        return len(self.listImagePaths)


def get_dataLoaderVAE(dataRoot, transformSequence, labelled=500, batch_size=8):
    txtFilePath = 'Dataset'
    pathDirData = dataRoot

    pathFileTrain_L = txtFilePath + '/train_' + str(labelled) + '.txt'
    pathFileTrain_U = txtFilePath + '/train_' + str(labelled) + '_unlab.txt'
    validation = txtFilePath + '/train_500_val_5000.txt'
    test = txtFilePath + '/train_500_test_10000.txt'

    datasetTrain_L = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain_L,
                                      transform=transformSequence)
    datasetTrain_U = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain_U,
                                      transform=transformSequence)
    datasetVal = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=validation,
                                  transform=transformSequence)
    datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=test,
                                   transform=transformSequence)

    dataLoaderTrain_L = DataLoader(dataset=datasetTrain_L, batch_size=batch_size, shuffle=True, num_workers=0,
                                   pin_memory=True)
    dataLoaderTrain_U = DataLoader(dataset=datasetTrain_U, batch_size=batch_size, shuffle=True, num_workers=0,
                                   pin_memory=True)
    dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=True, num_workers=0,
                                pin_memory=True)

    return dataLoaderTrain_L, dataLoaderTrain_U, dataLoaderVal, dataLoaderTest


def get_dataLoaderVAEEnsemble(dataRoot, labelled=500, batch_size=8):
    txtFilePath = 'Dataset'
    pathDirData = dataRoot

    pathFileTrain_L = txtFilePath + '/train_' + str(labelled) + '.txt'
    pathFileTrain_U = txtFilePath + '/train_' + str(labelled) + '_unlab.txt'
    validation = txtFilePath + '/train_500_val_5000.txt'
    test = txtFilePath + '/train_500_test_10000.txt'

    # Trained latent representation
    trainU = pathDirData + '/train_' + str(labelled) + '_unlab_vae.pkl'
    trainL = pathDirData + '/train_' + str(labelled) + '_vae.pkl'
    val_L = pathDirData + '/train_500_val_5000_vae.pkl'
    test_L = pathDirData + '/train_500_test_10000_vae.pkl'

    datasetTrain = DatasetGenerator_VAE_Ensemble(path=[trainL, trainU], textFile=[pathFileTrain_L, pathFileTrain_U])
    datasetTest = DatasetGenerator_VAE_Ensemble(path=[test_L], textFile=[test])

    #Note: Shuffle for train dataset should be set to False
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=True, num_workers=0,
                                pin_memory=True)

    return dataLoaderTrain, dataLoaderTest