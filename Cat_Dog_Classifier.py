# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
import random
import os
from keras.preprocessing import image
import numpy as np
from keras.regularizers import l2
import matplotlib.pyplot as plt



class Cat_Dog():
    def __init__(self):
        # Initialising the CNN
        self.classifier = Sequential()

        #first layer  Step 1 - Convolution
        self.classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', kernel_initializer='he_uniform'))
        # Step 2 - Pooling
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))


        # Adding a second convolutional layer
        self.classifier.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_uniform',bias_regularizer=l2(0.1)))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        self.classifier.add(Dropout(0.2))


        # Adding a third convolutional layer 76 to 74
        self.classifier.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_uniform')) #sigmoid=62
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        self.classifier.add(Dropout(0.2))

        # Step 3 - Flattening
        self.classifier.add(Flatten())

        # Step 4 - Full connection
        self.classifier.add(Dense(units = 128, activation = 'relu', kernel_initializer='he_uniform'))
        self.classifier.add(Dropout(0.5))  
        self.classifier.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Part 2 - Fitting the CNN to the images


        self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,rotation_range=30, brightness_range=[0.2,1.0],horizontal_flip = True)

        self.test_datagen = ImageDataGenerator(rescale = 1./255,)

        print('Training Set:-')
        self.training_set = self.train_datagen.flow_from_directory('dataset/training_set',
                                                         target_size = (64, 64),
                                                         batch_size = 8,
                                                         class_mode = 'binary')

        print('Test Set:-')
        self.test_set = self.test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 8,
                                                    class_mode = 'binary')


    def train_model(self):
        self.trained = self.classifier.fit_generator(self.training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = self.test_set,
                         validation_steps = 2000)

        # Getting summary
        summary = self.trained.history
        print(summary)
        x = str(round(self.classifier.evaluate_generator(self.test_set)[1]*100,3))+'%'
        self.classifier.save("Saved Model/Cat_Dog_model_Accuracy = "+x+" .h5")



    def load(self):
        self.classifier.load_weights('Saved Model/Cat_Dog_model_Accuracy = 86.6% .h5')
        print('accuracy : ' + str(round(self.classifier.evaluate_generator(self.test_set)[1]*100,3))+'%')


    def visualise(self):
        #print('accuracy : ' + str(round(self.classifier.evaluate_generator(self.test_set)[1]*100,3))+'%')

        #visualise
        self.act_list = [1]*1000
        self.pred_list_Dog = []
        self.pred_list_Cat = []

        for i in range(4001,5000):
            file = './dataset/test_set/dogs/dog.'+str(i)+'.jpg'
            img = image.load_img(file, target_size=(64, 64))
            img = image.img_to_array(img.convert('RGB'))
            img = np.expand_dims(img, axis=0)
            pred = self.classifier.predict_classes(img)[0][0]
            self.pred_list_Dog.append(pred)

        for i in range(4001,5000):
            file = './dataset/test_set/cats/cat.' + str(i) + '.jpg'
            img = image.load_img(file, target_size=(64, 64))
            img = image.img_to_array(img.convert('RGB'))
            img = np.expand_dims(img, axis=0)
            pred = self.classifier.predict_classes(img)[0][0]
            self.pred_list_Cat.append(pred)


        #visualise
        from matplotlib.patches import Rectangle
        #prediction histogram Test set
        # create legend
        labels = ["Dog","Cat"]
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['#EC4D37','#1D1B1B']]
        plt.legend(handles, labels)
        #histogram
        ticks = [0.17,0.835]
        label1 = ['Cat','Dog']
        plt.hist([self.pred_list_Cat,self.pred_list_Dog],bins=3,color=['#1D1B1B','#EC4D37'])
        plt.xticks(ticks,label1)
        plt.title('CNN prediction on test images')
        plt.xlabel('Actual Class')
        plt.ylabel('Frequency')
        plt.show()

        #Confusion Matrix
        import seaborn as sns
        self.confusion_matrix =[]
        x = [self.pred_list_Dog.count(1),self.pred_list_Dog.count(0)]
        y = [self.pred_list_Cat.count(1),self.pred_list_Cat.count(0)]
        # create legend
        labels = ["Wrong Prediction","Correct Prediction"]
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['#293250','#FFD55A']]
        plt.legend(handles, labels)
        #heatmap
        ax = plt.axes()
        ax.set_title('Confusion Matrix')
        self.confusion_matrix.append(x)
        self.confusion_matrix.append(y)
        sns.set(font_scale=1.8)
        sns.heatmap(self.confusion_matrix,xticklabels=['Dog','Cat'] ,yticklabels=['Dog','Cat'],cmap=['#293250','#FFD55A'],ax=ax,annot=True,fmt='d', cbar=False)
        ax.set_title('Confusion Matrix')
        plt.xlabel('Actual Class')
        plt.ylabel('Predicted Class')
        plt.show()



    def detect(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        print(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        img = image.img_to_array(img.convert('RGB'))
        img = np.expand_dims(img,axis=0)
        pred = self.classifier.predict_classes(img)[0][0] # dog - 1 cat - 0
        print(pred)


if __name__=='__main__':
    model = Cat_Dog()
    model.train_model()
    model.visualise()
