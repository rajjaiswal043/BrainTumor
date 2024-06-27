import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score ,accuracy_score,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

mri_images='BrainTumor/data'

no_tumor_mri_images=os.listdir(mri_images+ '/no/')
yes_tumor_mri_images=os.listdir(mri_images+ '/yes/')

# print(no_tumor_mri_images) #it print all the files and which is open by opencv documemtation
# path='no0.jpg' #providing the path of the file like we have seen in the previous print that all file are uploaded 
# print(path.split('.')[1]) #tell the type of file like this file type is jpg type coz its a image 
INPUT_SIZE=64
dataset=[]
label=[]

for i,image_name in enumerate(no_tumor_mri_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(mri_images+ '/no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i,image_name in enumerate(yes_tumor_mri_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(mri_images+ '/yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
dataset=np.array(dataset)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2)

# print(x_train.shape) #it reshape =  n, image_width,image_height,n_channel)
# print(y_train.shape) #it reshape =  n, image_width,image_height,n_channel)

# print(x_test.shape)
# print(y_test.shape)
x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)


# model building

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(INPUT_SIZE,INPUT_SIZE,3)))    #    64,64,3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))    
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))    
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(Dense(1))   #binary_crossentropy=1 ,sigmoid
model.add(Dense(2))     #catagorical cross entryopy = 2 ,softmax
# model.add(Activation('sigmoid'))    #binary_crossentropy=1 ,sigmoid
model.add(Activation('softmax'))    #categorical cross entryopy = 2 ,softmax


# model.compile(loss='binary_crossentropy',optimizer='adam',metrices=['accuracy'])    #binary_crossentropy=1 ,sigmoid
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])    #categorical cross entryopy = 2 ,softmax

history = model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=20,validation_data=(x_test,y_test),shuffle=False)

# model.save('Braintumor10Epochs.h5')
model.save('BrainTumor/Braintumor.keras')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='brown')
plt.plot(history.history['val_loss'], label='Validation Loss', color='yellow')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='brown')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='yellow')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

f1 = f1_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
print("F1 Score: {:.2f}".format(f1))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

accuracy= accuracy_score(y_true, y_pred_classes)
roc_auc = roc_auc_score(y_true, y_pred_classes)



print("accuracy: {:.2f}".format(accuracy))
print("roc_auc: {:.2f}".format(roc_auc))

confusion_mat = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", confusion_mat)

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
y_pred_rf = rf_model.predict(x_test.reshape(x_test.shape[0], -1))
rf_accuracy = rf_model.score(x_test.reshape(x_test.shape[0], -1), y_test)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted')  
rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

print("Random forest accuracy: {:.4f}".format(rf_accuracy))
print("Random forest precision: {:.4f}".format(rf_precision))
print("Random forest recall: {:.4f}".format(rf_recall))
print("Random forest F1-score: {:.4f}".format(rf_f1))

# decision tree classification
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
y_pred_dt = dt_model.predict(x_test.reshape(x_test.shape[0], -1))
dt_accuracy = dt_model.score(x_test.reshape(x_test.shape[0], -1), y_test)
dt_precision = precision_score(y_test, y_pred_dt, average='macro')  
dt_recall = recall_score(y_test, y_pred_dt, average='macro')
dt_f1 = f1_score(y_test, y_pred_dt, average='macro')


print("Decision tree accuracy: {:.4f}".format(dt_accuracy))
print("Decision tree precision: {:.4f}".format(dt_precision))
print("Decision tree recall: {:.4f}".format(dt_recall))
print("Decision tree F1-score: {:.4f}".format(dt_f1))

# svm model
# svm_model = SVC(kernel='linear', C=1)
# svm_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
# svm_accuracy = svm_model.score(x_test.reshape(x_test.shape[0], -1), y_test)

# naive bayes
# nb_model = GaussianNB()
# nb_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
# nb_accuracy = nb_model.score(x_test.reshape(x_test.shape[0], -1), y_test)

# logisticregression
# lr_model = LogisticRegression(max_iter=1000, random_state=0)
# lr_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
# lr_accuracy = lr_model.score(x_test.reshape(x_test.shape[0], -1), y_test)

# print("SVC : {:.2f}".format(svm_accuracy))
# print("Navie Bayes: {:.2f}".format(nb_accuracy))
# print("Logistic Regression: {:.2f}".format(lr_accuracy))