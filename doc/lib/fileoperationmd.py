# version 1
import os
def getFilesFromPath(path):
    return os.listdir(path)
def readTxtFromFile(path,filename):
    f=open(path+"/"+filename)
    return f.read()
