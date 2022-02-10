# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 01:35:03 2021

@author: SEREF
"""
from PyQt5.QtWidgets import*
from design_python import Ui_MainWindow
import tkinter
from tkinter import filedialog
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from PyQt5 import QtGui
import seaborn as sn
import pandas as pd

class design(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("DEEPPY")
        
        self.ui.spinBox_batch_size.setValue(32)
        self.ui.spinBox_target_size.setValue(250)
        self.ui.spinBox_epochs.setValue(2)
        
        self.ui.pushButton_model_selection.clicked.connect(self.model)
        self.ui.pushButton_data_selection.clicked.connect(self.data)
        self.ui.pushButton_train.clicked.connect(self.train)
        self.ui.comboBox.currentIndexChanged['int'].connect(self.properties)
        self.ui.comboBox_5.currentIndexChanged['int'].connect(self.data_graph)
        self.ui.comboBox_4.currentIndexChanged['int'].connect(self.result_graph)
        self.ui.pushButton_browse.clicked.connect(self.browse)
        self.ui.horizontalSlider.valueChanged.connect(self.slider)
        self.ui.pushButton_import.clicked.connect(self.import_data)
        self.ui.pushButton_train_2.clicked.connect(self.training)
        
        self.ui.pushButton_data_selection.setVisible(False)
        self.ui.pushButton_train.setVisible(False)
    def model(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    def data(self):
        self.ui.stackedWidget.setCurrentIndex(1)
    def train(self):
        self.ui.stackedWidget.setCurrentIndex(2)
    def slider(self):
        self.ui.lineEdit_10.setText(str(self.ui.horizontalSlider.value()))
        self.validation=self.ui.horizontalSlider.value()
    def train_2(self,model):
        for i, layer in enumerate(model.layers):
            print(i, layer.name)
        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        preds = tf.keras.layers.Dense(5, activation ='softmax')(x)
        
        #We can see that the final layer has 2 neurons as the output since we're classifying 5 classes. 
        #Now we're ready to create our own network, which consists of the base model and the output, which is our preds:
        
        
        model = tf.keras.models.Model(inputs=model.input, outputs=preds)
        print(model.summary())
        for i, layer in enumerate(model.layers):
            print(i, layer.name)
        if self.ui.checkBox_transfer.isChecked():
            for layer in model.layers[:19]:
                layer.trainable = False
        #Then for layer global_average_pooling2d and up we want these layers to be trainable:        
            for layer in model.layers[19:]:
                layer.trainable = True
        if self.ui.checkBox_deep.isChecked():
                for layer in model.layers[:19]:
                    layer.trainable = True
            #Then for layer global_average_pooling2d and up we want these layers to be trainable:        
                for layer in model.layers[19:]:
                    layer.trainable = True
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        epochs = self.ui.spinBox_epochs.value()
        batch_size = 30
        self.history = model.fit_generator(
            self.train_data_dir,
            steps_per_epoch = self.train_data_dir.samples // batch_size, 
            validation_data = self.validation_dir, 
            validation_steps = self.validation_dir.samples // batch_size,
            epochs = epochs,
            )
        print("eğitim bitti")
        
        print("Predictions hesaplaniyor")
        self.Y_pred = model.predict(self.validation_dir, self.validation_dir.samples // batch_size+1)
        self.y_pred = np.argmax(self.Y_pred, axis=1)
        self.cm = confusion_matrix(self.validation_dir.classes, self.y_pred)
        print(self.cm)
        labels = ["A", "B", "C", "D", "E"]
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title("Confusion Matrix")
        sns.set(font_scale=1.4)
        ax = sns.heatmap(self.cm, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set(ylabel="True Label", xlabel="Predicted Label")
        plt.savefig("confusion.png", bbox_inches='tight', dpi=300)
        plt.close()
        print("Hesaplandi")
             
    def training(self):
        if self.mod==1:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.VGG16(weights='imagenet', include_top = False)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.VGG16(weights=None, include_top = False)
                self.train_2(model)
        if self.mod==2:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.VGG19(weights='imagenet', include_top = False)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.VGG19(weights=None, include_top = False)
                self.train_2(model)
        if self.mod==3:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.Xception(include_top=True,weights="imagenet",input_tensor=None,input_shape=None,
    pooling=None,classifier_activation="softmax")
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.Xception(include_top=False,weights=None,input_tensor=None,input_shape=None,
    pooling=None,classifier_activation="softmax")
                self.train_2(model)
        if self.mod==4:
            if self.ui.checkBox_transfer.isChecked():
                model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top = False)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model = tf.keras.applications.ResNet50V2(weights=None, include_top = False)
                self.train_2(model)
        if self.mod==5:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.ResNet101(include_top=True,weights="imagenet",input_tensor=None,input_shape=None,
            pooling=None,**kwargs)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.ResNet101(include_top=False,weights=None,input_tensor=None,input_shape=None,
            pooling=None)
                self.train_2(model)
        if self.mod==6:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.MobileNet(input_shape=None, alpha=1.0,depth_multiplier=1,dropout=0.001,include_top=True,
            weights="imagenet",input_tensor=None,pooling=None,classifier_activation="softmax", **kwargs)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.MobileNet(input_shape=None, alpha=1.0,depth_multiplier=1,dropout=0.001,include_top=False,
            weights=None,input_tensor=None,pooling=None,classifier_activation="softmax")
                self.train_2(model)
        if self.mod==7:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.InceptionV3(include_top=True,weights="imagenet",input_tensor=None,input_shape=None,pooling=None,
    classifier_activation="softmax")
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.InceptionV3(include_top=False,weights=None,input_tensor=None,input_shape=None,pooling=None,
    classifier_activation="softmax")
                self.train_2(model)
        if self.mod==8:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.NASNetLarge(input_shape=None,include_top=True,weights="imagenet",input_tensor=None,
    pooling=None)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.NASNetLarge(input_shape=None,include_top=False,weights=None,input_tensor=None,
    pooling=None)
                self.train_2(model)
        if self.mod==9:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.EfficientNetB0(include_top=True, weights="imagenet",input_tensor=None,input_shape=None,
    pooling=None,classifier_activation="softmax",**kwargs)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.EfficientNetB0(include_top=False, weights=None,input_tensor=None,input_shape=None,
    pooling=None,classifier_activation="softmax")
                self.train_2(model)
        if self.mod==10:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.ResNet152(include_top=True, weights="imagenet",input_tensor=None,input_shape=None,pooling=None,
    **kwargs)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.ResNet152(include_top=False, weights=None,input_tensor=None,input_shape=None,pooling=None)
                self.train_2(model)
        if self.mod==11:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.MobileNetV2(input_shape=None,alpha=1.0,include_top=True,weights="imagenet",input_tensor=None,
    pooling=None, classifier_activation="softmax", **kwargs)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.MobileNetV2(input_shape=None,alpha=1.0,include_top=False,weights=None,input_tensor=None,
    pooling=None, classifier_activation="softmax")
                self.train_2(model)
        if self.mod==12:
            if self.ui.checkBox_transfer.isChecked():
                model=tf.keras.applications.DenseNet121(include_top=True,weights="imagenet",input_tensor=None,input_shape=None,
    pooling=None)
                self.train_2(model)
            if self.ui.checkBox_deep.isChecked():
                model=tf.keras.applications.DenseNet121(include_top=False,weights=None,input_tensor=None,input_shape=None,
    pooling=None)
                self.train_2(model)
       
    def result_graph(self,current_index):
        if current_index==1:
            self.ui.widget.canvas.axes.clear()
            self.ui.widget.canvas.axes.plot(self.history.history['loss'])
            self.ui.widget.canvas.axes.plot(self.history.history['val_loss'])
            self.ui.widget.canvas.axes.legend(['train', 'test'], loc='upper left')
            self.ui.widget.canvas.axes.set_xlabel("epochs")
            self.ui.widget.canvas.axes.set_ylabel("Loss")
            self.ui.widget.canvas.axes.set_title("Model Loss")
            self.ui.widget.canvas.draw()
        if current_index==2:
            self.ui.widget.canvas.axes.clear()
            self.ui.widget.canvas.axes.plot(self.history.history['accuracy'])
            self.ui.widget.canvas.axes.plot(self.history.history['val_accuracy'])
            self.ui.widget.canvas.axes.legend(['train', 'test'], loc='upper left')
            self.ui.widget.canvas.axes.set_xlabel("epochs")
            self.ui.widget.canvas.axes.set_ylabel("Accuracys")
            self.ui.widget.canvas.axes.set_title("Model Accuracy")
            self.ui.widget.canvas.draw()
        if current_index==3:
            image= plt.imread('confusion.png')
            self.ui.widget.canvas.axes.clear()
            self.ui.widget.canvas.axes.imshow(image)
            self.ui.widget.canvas.draw()
    def data_graph(self,current_index):
        if current_index ==1:
            self.ui.lineEdit_totalimage_2.setText(str(self.total_image_train))
            self.ui.lineEdit_totalclass_2.setText(str(len(self.category_names)))
            self.ui.lineEdit_mostclass.setText(self.train_most)
            self.ui.lineEdit_mostpicture.setText(str(max(self.train_number)))
            self.ui.lineEdit_lessclass.setText(self.train_less)
            self.ui.lineEdit_lesspicture.setText(str(min(self.train_number)))
            self.ui.MplWidget.canvas.axes.clear()
            self.ui.MplWidget.canvas.axes.bar(self.category_names,self.train_number,color="black")
            self.ui.MplWidget.canvas.axes.set_xlabel("Classes")
            self.ui.MplWidget.canvas.axes.set_ylabel("Number of Images")
            self.ui.MplWidget.canvas.axes.set_title("Train Dataset")
            self.ui.MplWidget.canvas.draw()
        if current_index ==2:
            self.ui.lineEdit_totalimage_2.setText(str(self.total_image_validation))
            self.ui.lineEdit_totalclass_2.setText(str(len(self.category_names)))
            self.ui.lineEdit_mostclass.setText(self.validation_most)
            self.ui.lineEdit_mostpicture.setText(str(max(self.validation_number)))
            self.ui.lineEdit_lessclass.setText(self.validation_less)
            self.ui.lineEdit_lesspicture.setText(str(min(self.validation_number)))
            self.ui.MplWidget.canvas.axes.clear()
            self.ui.MplWidget.canvas.axes.bar(self.category_names,self.validation_number,color="blue")
            self.ui.MplWidget.canvas.axes.set_xlabel("Classes")
            self.ui.MplWidget.canvas.axes.set_ylabel("Number of Images")
            self.ui.MplWidget.canvas.axes.set_title("Validation Dataset")
            self.ui.MplWidget.canvas.draw()
        
        
        
    def import_data(self):  
        batchsize=self.ui.spinBox_batch_size.value()
        targetsize=self.ui.spinBox_target_size.value()
        data_dir = self.tempdir
        image_generator = ImageDataGenerator(rescale=1/255, validation_split=self.validation/100)    
        
        if self.ui.checkBox_3.isChecked():
            self.train_data_dir = image_generator.flow_from_directory(batch_size=batchsize,
                                                             directory=data_dir,
                                                             shuffle=True,
                                                             target_size=(targetsize,targetsize), 
                                                             subset="training",
                                                             class_mode='categorical')
            
            self.validation_dir = image_generator.flow_from_directory(batch_size=batchsize,
                                                             directory=data_dir,
                                                             shuffle=True,
                                                             target_size=(targetsize,targetsize), 
                                                             subset="validation",
                                                             class_mode='categorical')
        else:
            self.train_data_dir = image_generator.flow_from_directory(batch_size=batchsize,
                                                             directory=data_dir,
                                                             shuffle=False,
                                                             target_size=(targetsize,targetsize), 
                                                             subset="training",
                                                             class_mode='categorical')
            
            self.validation_dir = image_generator.flow_from_directory(batch_size=batchsize,
                                                             directory=data_dir,
                                                             shuffle=False,
                                                             target_size=(targetsize,targetsize), 
                                                             subset="validation",
                                                             class_mode='categorical')
        self.category_names=sorted(os.listdir(self.tempdir))
        self.train_number=[]
        self.validation_number=[]
        for i in range(len(self.category_names)):
            self.train_number.append(0)
            self.validation_number.append(0)
        for i in range(int(242/32)):
            try:
                for j in range(len(self.train_data_dir[i][1])):
                    self.train_number[list(self.train_data_dir[i][1][j]).index(1)]+=1
            except:
                pass
        for i in range(int(242/32)):
            try:
                for j in range(len(self.validation_dir[i][1])):
                    self.validation_number[list(self.validation_dir[i][1][j]).index(1)]+=1
            except:
                pass
        self.total_image_train=0
        self.total_image_validation=0
        for i in range(len(self.train_number)):
            self.total_image_train+=self.train_number[i]
        for i in range(len(self.validation_number)):
            self.total_image_validation+=self.validation_number[i]
        train_most_image=self.train_number.index(max(self.train_number))
        train_less_image=self.train_number.index(min(self.train_number))
        self.train_most=self.category_names[train_most_image]
        self.train_less=self.category_names[train_less_image]
        
        validation_most_image=self.validation_number.index(max(self.validation_number))
        validation_less_image=self.validation_number.index(min(self.validation_number))
        self.validation_most=self.category_names[validation_most_image]
        self.validation_less=self.category_names[validation_less_image]
        print("bitti")
        self.ui.pushButton_train.setVisible(True)
    def browse(self):
        root = tkinter.Tk()
        root.withdraw() #use to hide tkinter window
        currdir = os.getcwd()
        self.tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')
        self.ui.lineEdit_5.setText(self.tempdir)
    def properties(self,current_index):
        if current_index==1:            
            self.ui.lineEdit_layer.setText("24")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=1
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap("1-vgg16.png"))
        if current_index ==2:
            self.ui.lineEdit_layer.setText("27")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=2
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap("2-vgg19.png"))
        if current_index ==3:
            self.ui.lineEdit_layer.setText("137")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=3
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap("3-xception.png"))
        if current_index ==4:
            self.ui.lineEdit_layer.setText("195")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=4
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap("4-resnet50.png"))
        if current_index ==5:
            self.ui.lineEdit_layer.setText("")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=5
            self.ui.label_8.setPixmap(QtGui.QPixmap("5-resnet101.png"))
            self.ui.pushButton_data_selection.setVisible(True)
        if current_index ==6:
            self.ui.lineEdit_layer.setText("91")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=6
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap(""))
        if current_index ==7:
            self.ui.lineEdit_layer.setText("316")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=7
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap("10-inceptionv3.png"))
        if current_index ==8:
            self.ui.lineEdit_layer.setText("1044")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=8
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap(""))
        if current_index ==9:
            self.ui.lineEdit_layer.setText("242")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=9
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap(""))
        if current_index ==10:
            self.ui.lineEdit_layer.setText("520")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=10
            self.ui.label_8.setPixmap(QtGui.QPixmap(""))
            self.ui.pushButton_data_selection.setVisible(True)
        if current_index ==11:
            self.ui.lineEdit_layer.setText("159")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=11
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap("13-mobilenetv2.png"))
        if current_index ==12:
            self.ui.lineEdit_layer.setText("432")
            self.ui.lineEdit_input.setText("Image")
            self.ui.lineEdit_output.setText("Classification")
            self.mod=12
            self.ui.pushButton_data_selection.setVisible(True)
            self.ui.label_8.setPixmap(QtGui.QPixmap(""))
