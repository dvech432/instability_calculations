##### build an initial data pipeline, using the file with the 10k calc


### load in images and resize them
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import numpy as np

import os
folder_content=os.listdir(r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images')
#folder_content.sort()
import skimage.io as io
all_images = []
for i in range(0,len(folder_content)):
  filename = r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images\\' + folder_content[i]
  img = io.imread(filename,as_gray=True)
  all_images.append(img)
x_train = np.array(all_images)   
x_train = x_train.reshape(-1, 100, 100, 1)
 
#### get regression values and normalize them
filename = r'C:\Users\vechd\.spyder-py3\instability_calc\params.txt'
params=np.array(pd.read_csv(filename))

from scipy import stats
core_ani=params[:,2]/params[:,1]
core_ani=stats.zscore(core_ani)
params=stats.zscore(params,axis=0)

#### test train
# %%
### define network
from cnn_regression import cnn_regression
m=cnn_regression()

m.fit(x_train, np.reshape(params[:,1],(-1,1)), batch_size=100,epochs=200, verbose=1)

core_ani_pred=m.predict(x_train)

plt.scatter(core_ani_pred, np.reshape(params[:,1],(-1,1)))

# %%
io.imshow(x_train[129,:,:,:])
plt.show()