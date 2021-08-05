########### build a classifier that identifies stable/unstable VDFs

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import numpy as np

##################### Load in image files
import os
folder_content=os.listdir(r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images')
#folder_content.sort()
import skimage.io as io
all_images = []
for i in range(0,len(folder_content)):
  filename = r'C:\Users\vechd\.spyder-py3\instability_calc\VDF_images\\' + folder_content[i]
  img = io.imread(filename,as_gray=True)
  all_images.append(img)
  print(str(i))
all_images = np.array(all_images)   
all_images = all_images.reshape(-1, 100, 100, 1)
 
##################### Load in Y param

filename = r'C:\Users\vechd\.spyder-py3\instability_calc\plume.txt'
plume = pd.read_csv(filename)
case_id=np.reshape(np.array(plume['case']),(-1,1))
case_id=np.where(case_id>0)[0] # select unstable cases
plume2=np.array(plume)
plume_unstable=plume2[case_id,:] # param list for unstable cases

from scipy import stats
#plume_unstable_norm=stats.zscore(plume_unstable,axis=0)

plume_unstable[:,13]=np.log10(plume_unstable[:,13])

k_mag = np.sqrt(plume_unstable[:,10]**2 + plume_unstable[:,11]**2).T # magnitude of the k-vector

plume_unstable_norm=stats.zscore(plume_unstable,axis=0)

k_mag_norm=stats.zscore(np.log10(k_mag),axis=0)

all_images_unstable = all_images[case_id,:,:,:]
##################### split into train and test

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(all_images_unstable, plume_unstable_norm[:,13], test_size=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(all_images_unstable, k_mag_norm, test_size=0.15, random_state=42)
##################### model training

from cnn_regression import cnn_regression
m=cnn_regression()

m.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=100,epochs=60, verbose=1)

Y_pred=m.predict(X_test)

plt.scatter(Y_pred, y_test)
plt.xlabel('Predicted value')
plt.ylabel('Ground truth value')

from sklearn.metrics import mean_squared_error
errors = mean_squared_error(y_test, Y_pred, squared=False)
print(errors)