import cv2
import os
import csv
import numpy as np
from google.colab import drive
import math

def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(cv2.resize(img,(224,224)))
            #images.append(img)
            pos = filename.find(".")
            names.append(int(filename[:pos]))
    return [names,images]
	
def mount_data(folder):
  drive.mount('/content/drive')

  folder2 = folder + "/S0-0_0/Images"
  images_S0_0_0 = load_images_from_folder(folder2)
  folder2 = folder + "/S0-0_60/Images"
  images_S0_0_60 = load_images_from_folder(folder2)
  folder2 = folder + "/S0-0_120/Images"
  images_S0_0_120 = load_images_from_folder(folder2)
  pot = []
  fw = []
  with open(folder + "/harvest.txt") as tsv:
      for line in csv.reader(tsv, delimiter="\t"):
          if line[0] =='pot':
              continue
          pot.append(int(line[0]))
          fw.append(float(str.replace(line[2],',','.')))

  FW = []
  images = []
  for i in images_S0_0_0:
      images.append([i[0],i[1],np.array(np.zeros((512,512,3))),np.array(np.zeros((512,512,3)))])
      for i60 in images_S0_0_60:
          if i[0]==i60[0]:
              images[-1][2] = i60[1]
              break
      for i120 in images_S0_0_120:
          if i[0]==i120[0]:
              images[-1][3] = i120[1]
              break
	
  data = [images,FW]

  for p in images[0]:
      if p in pot:
          FW.append(fw[pot.index(p)])
  return data  

def Divide_in_classes(Y_orig,Y_orig2, interval):

  classes = math.ceil(max([max(Y_orig),max(Y_orig2)])/interval)
  n = Y_orig.shape[0]
  Y_cl = np.zeros([n,classes])
  for y in range(n):
    Y_cl[y,math.floor(Y_orig[y] / interval)] = 1

  n = Y_orig2.shape[0]
  Y_cl2 = np.zeros([n,classes])
  for y in range(n):
    Y_cl2[y,math.floor(Y_orig2[y] / interval)] = 1

  return Y_cl,Y_cl2,classes
  
