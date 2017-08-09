
# coding: utf-8

# # Invasive Speices Monitoring (Kaggle) with Keras

# In[ ]:


import pandas as pd
import numpy as np
import os, shutil
import glob
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications import Xception
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers


# ### Read labels

# In[ ]:


train_dir = "./train"
val_dir = "./val"
test_dir = "./test"

# Load training labels
train_labels = pd.read_csv("train_labels.csv")


# ### Create directory structure

# In[ ]:


train_labels['name'] = train_labels['name'].apply(lambda x: str(x)+".jpg")
train_labels.set_index(train_labels['name'],inplace=True)


# In[ ]:


# Training images (all but last 500)
move_from = "./train/No_Invasive/"
move_to = "./train/Invasive/"
files = os.listdir(move_from)
files.sort()
for f in files:    
    if train_labels.loc[f]['invasive']:        
        src = move_from+f
        dst = move_to+f
        shutil.move(src,dst)


# In[ ]:


# Validation images (last 500)
move_from = "./val/No_Invasive/"
move_to = "./val/Invasive/"
files = os.listdir(move_from)
files.sort()
for f in files:    
    if train_labels.loc[f]['invasive']:        
        src = move_from+f
        dst = move_to+f
        shutil.move(src,dst)


# ### CNN Architecture (use pre-trained network: Xception)

# #### Iterate to find the optimal architecture

# In[ ]:


img_rows = 299
img_cols = 299
batch_size = 16
num_nodes = [512,1024,2048,4096,8192]
num_layers = [1,2,3]
dropout_prob = [0.2,0.3,0.4,0.5]

for layers in num_layers:
    for nodes in num_nodes:
        for p in dropout_prob:

            model_param = str(layers) + "-" + str(nodes) + "-" + str(p)
            print("**********************************************************************************************")
            print(model_param)
            print("**********************************************************************************************")
            
            xception_base = Xception(input_shape=(img_cols, img_rows, 3), weights='imagenet', include_top=False)
            for layer in xception_base.layers:
                layer.trainable = False

            dense_model = Sequential()
            dense_model.add(GlobalAveragePooling2D(input_shape=xception_base.output_shape[1:]))        

            for i in range(0,layers):
                dense_model.add(Dense(nodes, activation='relu'))
                dense_model.add(Dropout(p))
                dense_model.add(BatchNormalization())

            dense_model.add(Dense(1, activation='sigmoid'))    

            final_model = Model(inputs=xception_base.input, outputs=dense_model(xception_base.output))
            final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

            train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,
                                               zoom_range=0.2,horizontal_flip=True)

            train_generator = train_datagen.flow_from_directory(train_dir,target_size=(img_cols, img_rows),batch_size=batch_size,
                                                                class_mode='binary')

            val_datagen = ImageDataGenerator(rescale=1.0/255)

            val_generator = val_datagen.flow_from_directory(val_dir,target_size=(img_cols, img_rows),batch_size=batch_size,
                                                                class_mode='binary')

            checkpointer = [ModelCheckpoint(filepath=model_param +'_invasive_xception.model.best.hdf5',verbose=1, 
                                            save_best_only=True),
                            EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

            final_model.fit_generator(train_generator,samples_per_epoch=1795//batch_size,
                                      epochs=10,validation_data=val_generator,nb_val_samples=500//batch_size,
                                      callbacks=checkpointer)
            K.clear_session()


# ### Final model architecture

# In[ ]:


img_rows = 299
img_cols = 299
batch_size = 16
xception_base_layer = 126

xception_base = Xception(input_shape=(img_cols, img_rows, 3), weights='imagenet', include_top=False)

dense_model = Sequential()
dense_model.add(GlobalAveragePooling2D(input_shape=xception_base.output_shape[1:]))        

dense_model.add(Dense(4096, activation='relu'))
dense_model.add(Dropout(0.4))
dense_model.add(BatchNormalization())

dense_model.add(Dense(4096, activation='relu'))
dense_model.add(Dropout(0.4))
dense_model.add(BatchNormalization())

dense_model.add(Dense(1, activation='sigmoid'))    

final_model = Model(inputs=xception_base.input, outputs=dense_model(xception_base.output))
final_model.load_weights('2-1024-0.3_invasive_xception.model.best.hdf5')

for layer in final_model.layers[:xception_base_layer]:
    layer.trainable = False
for layer in final_model.layers[xception_base_layer:]:
    layer.trainable = True

final_model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

checkpointer = [ModelCheckpoint(filepath='invasive_xception.model.best.hdf5',monitor='val_acc',verbose=1, 
                                save_best_only=True),
                EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

final_model.fit_generator(train_generator,samples_per_epoch=1795,
                          epochs=50,validation_data=val_generator,nb_val_samples=500,
                          callbacks=checkpointer)


# ### Predictions

# In[ ]:


img_rows = 299
img_cols = 299
batch_size = 1
xception_base_layer = 126
test_samples = 1531

# Load test images
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(img_cols, img_rows),batch_size=batch_size,
                                                                class_mode='binary',shuffle=False)

# Define model
xception_base = Xception(input_shape=(img_cols, img_rows, 3), weights='imagenet', include_top=False)

dense_model = Sequential()
dense_model.add(GlobalAveragePooling2D(input_shape=xception_base.output_shape[1:]))        

dense_model.add(Dense(1024, activation='relu'))
dense_model.add(Dropout(0.3))
dense_model.add(BatchNormalization())

dense_model.add(Dense(1024, activation='relu'))
dense_model.add(Dropout(0.3))
dense_model.add(BatchNormalization())

dense_model.add(Dense(1, activation='sigmoid'))    

final_model = Model(inputs=xception_base.input, outputs=dense_model(xception_base.output))
final_model.load_weights('invasive_xception.model.best.hdf5')

final_model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

final_model.summary()

predictions = final_model.predict_generator(test_generator,steps=test_samples)


# ### Prepare submission file

# In[ ]:


image_num = [int(x.replace("unknown\\", "").replace(".jpg", "")) for x in test_generator.filenames]
results = pd.DataFrame({'name':image_num, 'invasive':predictions.reshape((test_samples,))})
results.to_csv("results.csv")

