# Convolutional Neural Network
# Part 1 - Building the CNN
import numpy as np
import matplotlib.pyplot as plt

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import regularizers


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# glorot_uniform is the default kernel_initializer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', kernel_initializer='he_normal',
                      kernel_regularizer= regularizers.l2(l=0.001)))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a 2nd convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_normal',
                      kernel_regularizer= regularizers.l2(l=0.001)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a 3rd convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_normal',
                      kernel_regularizer= regularizers.l2(l=0.001)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a 4rth convolutional layer
classifier.add(Conv2D(256, (3, 3), activation = 'relu', kernel_initializer='he_normal',
                      kernel_regularizer= regularizers.l2(l=0.001)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Adding a 5th convolutional layer
#classifier.add(Conv2D(512, (3, 3), activation = 'relu', kernel_initializer='he_normal',
#                      kernel_regularizer= regularizers.l2(l=0.001)))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 1, activation = 'sigmoid')) #binary class

#For a single-input model with 10 classes (categorical classification):
#classifier.add(Dense(units = 10, activation = 'sigmoid')) 
#classifier.add(Dropout(0.25))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                      rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = classifier.fit_generator(training_set,
                         steps_per_epoch = 689,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 689)

#saving model as json
classifier_json = classifier.to_json()
with open("classifier.json","w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights('weights_10.h5')


predictions = classifier.predict_generator(test_set, steps = 221) 
y_pred = np.round(predictions) #binary categories
y_test = test_set.classes
class_names = test_set.class_indices.keys()
#predictions = np.argmax(np.round(predictions), axis=1) #multiple categories


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

animals=['cats','dogs']
from sklearn.metrics import classification_report, accuracy_score
classification_metrics = classification_report(y_test, y_pred, target_names=animals)
print(classification_metrics)
print(accuracy_score(y_test, y_pred))


from keras.preprocessing import image
#import cv2
import glob
#images = [cv2.imread(file) for file in glob.glob('../NN/*jpg')]

count_cats = 0
count_dogs = 0
for file in glob.glob('../NN/data/*jpg'):
    print(file)
    test_image = image.load_img(file,target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = np.round(classifier.predict(test_image))
    
    if result == 0:
        pred = 'cat'
        count_cats += 1
    else:
        pred = 'dog'
        count_dogs  += 1
    print(pred)

print("cats : {}".format(count_cats))
print("dogs : {}".format(count_dogs))


#Image Augmentation
from keras.preprocessing import image
import glob
import numpy as np
datagen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
height_shift_range=0.1,shear_range=0.15, 
zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)

for file in glob.glob('../NN/dataset/test_set/cats/*jpg'):
    test_image = image.load_img(file,target_size=(64, 64), grayscale=True) #grayscale=True
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)

    datagen.fit(test_image)

    for x, val in zip(datagen.flow(test_image,save_to_dir='../NN/dataset/test_set/cats/', 
                               save_prefix='augmented',save_format='jpg'),
    range(0)) :
        pass



###################################
#load model and weights
from keras.models import model_from_json
json_file = open("classifier.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json) 
loaded_model.load_weights("weights_10.h5")

# compile the model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
test_datagen = ImageDataGenerator(rescale = 1./255,
                                      rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

predictions = loaded_model.predict_generator(test_set, steps = 221) 
y_pred = np.round(predictions) #binary categories
y_test = test_set.classes
class_names = test_set.class_indices.keys()

#loading from the saved model
classifier = loaded_model
########################################


#plotting training accuracy and loss. 
#plt.figure(figsize=(20,10))
#plt.subplot(1, 2, 1)
#plt.suptitle('Optimizer : Adam', fontsize=10)
#plt.ylabel('Loss', fontsize=16)
#plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.legend(loc='upper right')
#
#plt.subplot(1, 2, 2)
#plt.ylabel('Accuracy', fontsize=16)
#plt.plot(history.history['acc'], label='Training Accuracy')
#plt.plot(history.history['val_acc'], label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.show()
        
#import itertools
#def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
#    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    plt.figure(figsize=(10,10))
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="red" if cm[i, j] > thresh else "black")
#
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.tight_layout()
#
#plt.figure()
#plot_confusion_matrix(cm, classes=class_names, title='Normalized confusion matrix')
#plt.show()
