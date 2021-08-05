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
 
##################### Load in image labels

filename = r'C:\Users\vechd\.spyder-py3\instability_calc\plume.txt'
plume = pd.read_csv(filename)

label=np.array(plume['case'])
label= np.nan_to_num(label, nan=0)
label[label>0]=1

### new idea: exclude marginally unstable cases
#label=pd.DataFrame(pd.get_dummies(plume[plume.columns[13]]>10**-2.5)  )

### old method
label=pd.DataFrame(pd.get_dummies((label==1)))

label=np.array(label.drop(0,axis=1))

    
##################### split into train and test
rr=15000

from sklearn.model_selection import train_test_split

idx_train, idx_test= train_test_split(np.arange(0,rr), test_size=0.15, random_state=42)

X_train=all_images[idx_train,:,:,:]
X_test=all_images[idx_test,:,:,:]
y_train=label[idx_train]
y_test=label[idx_test]

plume_array=np.array(plume)
plume_train=plume_array[idx_train,:]
plume_test=plume_array[idx_test,:]
#X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(all_images[0:rr,:,:,:], label[0:rr], test_size=0.15, random_state=42)

##################### model training

# from cnn_classification import cnn_classification
# m=cnn_classification()

from gen_AlexNet import gen_AlexNet
m=gen_AlexNet()

m.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=100,epochs=200, verbose=1)

#m.save(r'C:\Users\vechd\.spyder-py3\instability_calc\classification_trained_v2')
#m.save(r'C:\Users\vechd\.spyder-py3\instability_calc\classification_trained')

from keras.models import load_model
m = load_model(r'C:\Users\vechd\.spyder-py3\instability_calc\classification_trained')

pred_labels=np.round(m.predict(X_test))
pred_score=(m.predict(X_test))
##################### plotting confusion matrix


from sklearn.metrics import classification_report
classification_report(y_test,pred_labels)


tp=np.where( (pred_labels==1) & (y_test==1) )[0]
fp=np.where( (pred_labels==1) & (y_test==0) )[0]

tn=np.where( (pred_labels==0) & (y_test==0) )[0]
fn=np.where( (pred_labels==0) & (y_test==1) )[0]

tp_param=np.nanmean(plume_array[tp,:],axis=0).T
fp_param=np.nanmean(plume_array[fp,:],axis=0).T
print(tp_param/fp_param)


# %%
p=13
A=plume_test[:,p]
A=A[~np.isnan(A)]

B=plume_test[fn,p]
B=B[~np.isnan(B)]

fig, ax = plt.subplots()
ax.hist(np.log10(A),bins=20,label='All '+ plume.columns[p])
ax.hist(np.log10(B),bins=20,label=plume.columns[p] +' for incorrectly classified cases')
leg = ax.legend()
plt.xlabel('Log10 ' + plume.columns[p])
# %%
##### excluding marginally stable cases

len(np.where( (pred_labels==1) & (y_test==1) & ( np.reshape(plume_test[:,p],(-1,1)) >10**-3  ) )[0]  )

# %%

pred_score=(m.predict(X_test))

p1=np.where((pred_score>0.66) | (pred_score<0.33))[0]

len(np.where(pred_labels[p1]==y_test[p1])[0])/len(p1)

# %% plotting predicted probability against gamma/Omega

p=13
A=plume_test[:,p]

plt.scatter(pred_score,np.log10(A))
plt.xlim((0,1))


fig, ax = plt.subplots()
ax.scatter(pred_score,np.log10(A),label=plume.columns[p])
plt.xlabel('Predicted probability that the VDF is unstable')
plt.ylabel('Log10 ' + plume.columns[p])