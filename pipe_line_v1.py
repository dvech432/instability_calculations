##### build an initial data pipeline, using the file with the 10k calc


### load in images and resize them
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd

import os
folder_content=os.listdir(r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images')

img_grid=[]
for i in range(0,len(folder_content)):
  filename = r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images\\' + folder_content[i]
  image = Image.open(filename)
  new_image = image.resize((100, 100))
  img_grid.append(new_image)

#### get regression values
filename = r'C:\Users\vechd\.spyder-py3\instability_calc\params.txt'
params=pd.read_csv(filename)

#### test train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(img_grid, params[:,0], test_size=0.15, random_state=42)


### define network
from cnn import cnn
m=cnn()