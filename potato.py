#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers
data=[]
label=[]


# In[108]:


files=os.listdir("PlantVillage")
files


# In[202]:


for labels in files:
    folder_path=os.path.join("PlantVillage",labels)
    for img_files in os.listdir(folder_path):
        img_path=os.path.join(folder_path,img_files)
        img=cv2.imread(img_path)
        img=cv2.resize(img,(256,256))
        data.append(img)
        label.append(labels)
label



# In[204]:


import
for i in range(9):
    plt.imshow(x[1000].astype("uint8"))
    plt.title([label[1000]])


# In[ ]:





# In[ ]:





# In[ ]:


import tensorflow as tf

dataset = tf.keras.utils.image_dataset_from_directory(
    "PlantVillage",
    image_size=(256, 256),
    batch_size=32
)


# In[ ]:


dataset


# In[156]:


class_names=dataset.class_names
class_names


# In[ ]:


plt.figure(figsize=(10,10))
for img,label in dataset.take(1):
    for i in range(9):
        plt.subplot(3,4,i+1)
        plt.imshow(img[i].numpy().astype("uint8"))
        plt.title(class_names[label[i]])


# In[ ]:


train_ds=dataset.take(54)


# In[ ]:


test_ds=dataset.skip(54)


# In[ ]:


val_ds=test_ds.take(6)
len(val_ds)


# In[ ]:


test_ds=test_ds.skip(6)


# In[ ]:


len(test_ds)


# In[ ]:


def get_split(dataset,train_split=0.8,test_split=0.1,val_split=0.1,shuffle=True,shuffle_size=1000):
    ds_size=len(dataset)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    train_ds=dataset.take(train_size)
    val_ds=dataset.skip(train_size).take(val_size)
    test_ds=dataset.skip(train_size).skip(val_size)
    return train_ds,val_ds,test_ds



# In[ ]:


train_ds,val_ds,test_ds=get_split(dataset)


# In[ ]:


len(train_ds)


# In[ ]:


len(test_ds)


# In[111]:


len(val_ds)


# In[112]:


train_ds=train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)


# In[119]:


resize_rescale=tf.keras.Sequential([
    layers.Resizing(256,256),
    layers.Rescaling(1./255)

])




# In[121]:


data_agumentaion=tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4)
])


# In[144]:


from tensorflow.keras import models
bs=32
ims=256,
cl=3
n_classes=3
input_shape=(bs,ims,ims,cl)
model=models.Sequential([
    resize_rescale,
    data_agumentaion,
    layers.Conv2D(32,(3,3), activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
     layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
     layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
     layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
     layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')
])



# In[145]:


model.summary()


# In[146]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(
    train_ds,epochs=50,
    batch_size=32,
    verbose=1,
    validation_data=val_ds)


# In[147]:


scores=model.evaluate(test_ds)


# In[148]:


scores


# In[150]:


history.history.keys()


# In[153]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(range(50), acc, label='Training Accuracy')
plt.plot(range(50), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# 🔹 Loss Plot
plt.subplot(1, 2, 2)
plt.plot(range(50), loss, label='Training Loss')
plt.plot(range(50), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()


# In[165]:


for image,labels in test_ds.take(1):
    first =image[0].numpy().astype("uint8")
    fisrt_l=labels[0].numpy()
    print("actual iamge")
    plt.imshow(first)
    print("acutal label",class_names[fisrt_l])
    prediction=model.predict(image)
    print("predeicted",class_names[np.argmax(prediction[0])])


# In[161]:


np.argmax([9.8805064e-01,1.1949349e-02,2.3186466e-14])


# In[166]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence


# In[176]:


plt.figure(figsize=(15,15))
for image,labels in test_ds:
    for i in range(8):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i].numpy().astype("uint8"))
        prediction,confidence=predict(model,image[i].numpy())
        actual_class=class_names[labels[i]]
        plt.title(f"Actual: {actual_class}\nPredicted: {prediction}\nConfidence: {confidence}%")
        plt.axis("off")


# In[197]:


import os
max([int(i)for i in os.listdir("./models")])+1
model_version=max([int(i)for i in os.listdir("./models")])+1
model.export(f"./models/{model_version}")


# In[ ]:




