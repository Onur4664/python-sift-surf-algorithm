from sklearn.metrics import plot_confusion_matrix
import os
import csv
import cv2
import pandas as pd
import numpy as np
from PyQt5.uic import loadUiType
from görüntü import Ui_Form
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QTimer,QTime,QAbstractTableModel
from PyQt5 import uic
from pandas import DataFrame
from keras.models import load_model
from skimage.feature import daisy
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import  MaxPooling2D,Convolution2D,Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import multilabel_confusion_matrix
from skimage.filters import threshold_otsu
from skimage import io,color
import os,shutil
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.feature import daisy 
from skimage import data
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from sklearn.decomposition import PCA
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from keras.layers import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from PIL import ImageTk, Image
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
from PyQt5.QtWidgets import QApplication, QTableView, QFileDialog,QMessageBox,QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pathlib import Path

#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#h,s,v1=cv2.split(images)
#images=v1
from sklearn.datasets import make_classification


class MainWindow(QWidget,Ui_Form):
    dataset_file_path = ""
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)  
        self.setupUi(self)  
        self.pushButton.clicked.connect(self.resimyukle)
        self.pushButton_2.clicked.connect(self.sift)
        self.pushButton_3.clicked.connect(self.surf)
        self.pushButton_4.clicked.connect(self.egitim)    
        self.pushButton_5.clicked.connect(self.holdout)
#        self.pushButton_6.clicked.connect(self.ROC)
        self.pushButton_7.clicked.connect(self.yukle)
#        self.tableWidget.clicked.connect(self.tıklama)
        self.pushButton_8.clicked.connect(self.CNN)
        self.pushButton_9.clicked.connect(self.tahminet)
#        self.pushButton_10.clicked.connect(self.kfold)
#        self.pushButton_7.clicked.connect(self.KNN)
#        self.pushButton_8.clicked.connect(self.RFC)
#        self.pushButton_9.clicked.connect(self.LR)
#        self.pushButton_10.clicked.connect(self.LR)
#        self.pushButton_11.clicked.connect(self.LR)
#        self.pushButton_12.clicked.connect(self.LR)
        self.comboBox_2.addItem("0.3")
        self.comboBox_2.addItem("0.4")
        self.comboBox_2.addItem("0.5")
        self.comboBox_2.addItem("0.6")
        self.comboBox.addItem("KNN")
        self.comboBox.addItem("Random Forest")
        self.comboBox.addItem("Logistic Regression")
        
    def resimyukle(self):
#        resım=QFileDialog.getExistingDirectory(self, 'Audio Files', "select Directory")
        self.resım = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.yukle=os.listdir(self.resım)
        print("yükle",self.yukle)
        print("resım",self.resım)
        self.yol=self.resım +'/'
        print("yol",self.yol)
        self.degerler=os.listdir(self.yol)
        self.labels=[]
        k=[]
        l=[]
        print("+",self.degerler)
        for i,deger in enumerate(self.degerler):
            k=[]            
            self.labels.append(deger)
            print("labels",self.labels)
#            self.klasor_deger=os.listdir(yol+deger)
            files = os.listdir(self.yol + deger)
            for i,j in enumerate(files):                
                k.append(j)
            l.append(k)
        self.yeni = DataFrame(l)
        print("yeni",self.yeni.values)                                        
        c=len(self.yeni.columns)
        r=len(self.yeni.values)
        self.tableWidget.setColumnCount(r)
        self.tableWidget.setRowCount(c)
        self.tableWidget.setHorizontalHeaderLabels(self.labels)
        for i,row in enumerate(self.yeni):           
            for j,cell in enumerate(self.yeni.values):
                 self.tableWidget.setItem(i,j, QtWidgets.QTableWidgetItem(str(cell[i])))
                 
    def sift(self): 
        a=[]
        b=[]
        c=-1
        if self.radioButton.isChecked():
            for label_no,directory in enumerate(self.yukle):
                files=os.listdir(self.yol+directory)
                c+=1
                for file_name in files:
                    img2= cv2.imread("flowers10/"+directory+"/"+file_name) 
                    
                    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)  
                    cv2.imwrite("./rgbsift/"+file_name,img2)
                    sift = cv2.xfeatures2d.SIFT_create()
                    keypoints_sift,descriptors=sift.detectAndCompute(img2,None)
                    img2 = cv2.drawKeypoints(img2, keypoints_sift, None)                                        
                    for key in keypoints_sift:
                        x=(int(key.pt[0]))                        
                        y=(int(key.pt[1]))                                         
                    for i in range(10):                                 
                        region=img2[abs((y-25)):abs((y+25)),abs((x-25)):abs((x+25))]                       
                        region=cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = daisy(region, step=90, radius=3, rings=2, histograms=5, orientations=5, visualize=True) 
#                        descs=descs.reshape(descs.shape[0],descs.shape[1]*descs.shape[2])
                        descs=resize(descs, (30, 30))                      
#                        descs=descs.flatten()                            
                        a.append(descs)                       
                        b.append(c)       
            
            self.X=np.array(a)
            print("shape",self.X.shape)            
            print(len(self.X))
            self.Y=np.array(b)
            print("yshape",self.Y.shape)   
            pickle.dump(self.X,open('./sift/X.pkl', 'wb'))
            pickle.dump(self.Y,open('./sift/Y.pkl', 'wb'))
            print(len(self.X))
            print("RGB, sift için işlem tamam")            
        
        
        if self.radioButton_2.isChecked():
            for label_no,directory in enumerate(self.yukle):
                files=os.listdir(self.yol+directory)
                c+=1
                for file_name in files:
                    img2= cv2.imread("flowers10/"+directory+"/"+file_name)
                    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                    cv2.imwrite(("./hsvsift/"+file_name),img2)
                    sift = cv2.xfeatures2d.SIFT_create()
                    keypoints_sift,descriptors=sift.detectAndCompute(img2,None)
                    img2 = cv2.drawKeypoints(img2, keypoints_sift, None)                              
                    for key in keypoints_sift:
                        x=(int(key.pt[0]))                       
                        y=(int(key.pt[1]))                                         
                    for i in range(10):                        
                        region=img2[abs((y-25)):abs((y+25)),abs((x-25)):abs((x+25))]
                        region=cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = daisy(region, step=90, radius=3, rings=2, histograms=5, orientations=5, visualize=True) 
                        descs=resize(descs, (30, 30))
                        a.append(descs)
                        b.append(c)       
             
            self.X=np.array(a)
            print(len(self.X))
            self.Y=np.array(b)
            pickle.dump(self.X,open('./sift/X.pkl', 'wb'))
            pickle.dump(self.Y,open('./sift/Y.pkl', 'wb'))
            print("HSV, sift için işlem tamam")    
        
        if self.radioButton_3.isChecked():
            for label_no,directory in enumerate(self.yukle):
                files=os.listdir(self.yol+directory)
                c+=1
                for file_name in files:
                    img2= cv2.imread("flowers10/"+directory+"/"+file_name)
                    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)
                    cv2.imwrite(("./ciesift/"+file_name),img2)
                    sift = cv2.xfeatures2d.SIFT_create()
                    keypoints_sift,descriptors=sift.detectAndCompute(img2,None)
                    img2 = cv2.drawKeypoints(img2, keypoints_sift, None)                              
                    for key in keypoints_sift:
                        x=(int(key.pt[0]))                       
                        y=(int(key.pt[1]))                                         
                    for i in range(10):                        
                        region=img2[abs((y-25)):abs((y+25)),abs((x-25)):abs((x+25))]
                        region=cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = daisy(region, step=90, radius=3, rings=2, histograms=5, orientations=5, visualize=True)    
                        descs=resize(descs, (30, 30))                        
                        a.append(descs)
                        b.append(c)       
                        
            self.X=np.array(a)
            self.Y=np.array(b)
            pickle.dump(self.X,open('./sift/X.pkl', 'wb'))
            pickle.dump(self.Y,open('./sift/Y.pkl', 'wb'))
            print("CİE,sift için işlem tamam")        
        
        
        
        
        
    def surf(self):  
        a=[]
        b=[]
        c=-1
        if self.radioButton.isChecked():
            for label_no,directory in enumerate(self.yukle):
                files=os.listdir(self.yol+directory)
                c+=1
                for file_name in files:
                    img2= cv2.imread("flowers10/"+directory+"/"+file_name)  
                    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
                    cv2.imwrite(("./rgbsurf/"+file_name),img2)                     
                    surf = cv2.xfeatures2d.SURF_create()
                    keypoints_surf,descriptors=surf.detectAndCompute(img2,None)
                    img2 = cv2.drawKeypoints(img2, keypoints_surf, None)
                                   
                    x,y=[],[]
                    for key in keypoints_surf:
                        x=(int(key.pt[0]))
                        y=(int(key.pt[1]))
                        
                    for i in range(10):                       
                        region=img2[abs((y-25)):abs((y+25)),abs((x-25)):abs((x+25))]
                        region=cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = daisy(region, step=90, radius=3, rings=2, histograms=5, orientations=5, visualize=True) 
                        descs=resize(descs, (30, 30))
                        a.append(descs)
                        b.append(c)  
                        
            self.X=np.array(a)
            self.Y=np.array(b)
            pickle.dump(self.X,open('./surf/X.pkl', 'wb'))
            pickle.dump(self.Y,open('./surf/Y.pkl', 'wb'))
            print("RGB,surf için işlem tamam") 
#            print("Sift için algoritmayı tamamladık ")
#            print("Surf için algoritmayı tamamladık ")
#            print("HSV için algoritmayı tamamladık ")
        if self.radioButton_2.isChecked():
            for label_no,directory in enumerate(self.yukle):
                files=os.listdir(self.yol+directory)
                c+=1
                for file_name in files:
                    img2= cv2.imread("flowers10/"+directory+"/"+file_name)
                    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV) 
                    cv2.imwrite(("./hsvsurf/"+file_name),img2)                              
                    surf = cv2.xfeatures2d.SURF_create()                    
                    keypoints_surf,descriptors=surf.detectAndCompute(img2,None)
                    img2 = cv2.drawKeypoints(img2, keypoints_surf, None)
                                  
                    x,y=[],[]
                    for key in keypoints_surf:
                        x=(int(key.pt[0]))
                        y=(int(key.pt[1]))                        
                    for i in range(10):
                        region=img2[abs((y-25)):abs((y+25)),abs((x-25)):abs((x+25))]
                        region=cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = daisy(region, step=90, radius=3, rings=2, histograms=5, orientations=5, visualize=True) 
                        descs=resize(descs, (30, 30))
                        a.append(descs)
                        b.append(c)
             
            self.X=np.array(a)
            self.Y=np.array(b)
            pickle.dump(self.X,open('./surf/X.pkl', 'wb'))
            pickle.dump(self.Y,open('./surf/Y.pkl', 'wb'))
            print("HSV,surf için işlem tamam")
                
        if self.radioButton_3.isChecked():
            for label_no,directory in enumerate(self.yukle):
                files=os.listdir(self.yol+directory)
                c+=1
                for file_name in files:
                    img2= cv2.imread("flowers10/"+directory+"/"+file_name)
                    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)                                 
                    surf = cv2.xfeatures2d.SURF_create()
                    keypoints_surf,descriptors=surf.detectAndCompute(img2,None)
                    img2 = cv2.drawKeypoints(img2, keypoints_surf, None)
                                  
                
                    for key in keypoints_surf:
                        x=(int(key.pt[0]))
                        y=(int(key.pt[1]))                        
                    for i in range(10):                        
                        region=img2[abs((y-25)):abs((y+25)),abs((x-25)):abs((x+25))]
                        region=cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = daisy(region, step=90, radius=3, rings=2, histograms=5, orientations=5, visualize=True) 
                        descs=resize(descs, (30, 30))
                        print("features",descs.shape)
                        a.append(descs)
                        b.append(c)
                        
                        
            self.X=np.array(a)
            self.Y=np.array(b)                
            pickle.dump(self.X,open('./surf/X.pkl', 'wb'))
            pickle.dump(self.Y,open('./surf/Y.pkl', 'wb'))
            print("CİE,surf için işlem tamam")        
            print(len(self.X))
            
    def holdout(self):
        self.degisken=float(self.comboBox_2.currentText())
        self.x_train, self.x_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,test_size=self.degisken, random_state=42)
        pickle.dump(self.x_train,open('./sift/x_train.pkl', 'wb'))    
        pickle.dump(self.x_test,open('./sift/x_test.pkl', 'wb'))  
        pickle.dump(self.y_train,open('./sift/y_train.pkl', 'wb'))  
        pickle.dump(self.y_test,open('./sift/y_test.pkl', 'wb')) 
        pickle.dump(self.x_train,open('./surf/x_train.pkl', 'wb'))    
        pickle.dump(self.x_test,open('./surf/x_test.pkl', 'wb'))  
        pickle.dump(self.y_train,open('./surf/y_train.pkl', 'wb'))  
        pickle.dump(self.y_test,open('./surf/y_test.pkl', 'wb')) 
        self.x_train= self.x_train.reshape(self.x_train.shape[0], -1)       
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        print (self.x_train.shape,self.x_test.shape)
        print (self.y_train,self.y_test)
        self.textEdit_2.setText(str(self.x_train))
        self.textEdit_4.setText(str(self.x_test))
        self.textEdit_3.setText(str(self.y_train))
        self.textEdit_5.setText(str(self.y_test))
        
        
#    def kfold(self):
#         kf = KFold(n_splits=10) #Bölmeyi 5 olarak tanımladım 5 kez bölünecek
#         kf.get_n_splits(self.X)#içindeki bölme yinelemelerinin sayısını döndürür croos-validation...
#         KFold(n_splits=10, random_state=42, shuffle=True)
#         for train_x, test_x in kf.split(self.X):
#             print("TRAIN:", train_x, "TEST:", test_x)
#             x_train, x_test = self.X[train_x], self.X[test_x]
#             y_train, y_test = self.Y[train_x], self.Y[test_x]
#             pickle.dump(self.x_train,open('./sift/x_train.pkl', 'wb'))    
#             pickle.dump(self.x_test,open('./sift/x_test.pkl', 'wb'))  
#             pickle.dump(self.y_train,open('./sift/y_train.pkl', 'wb'))  
#             pickle.dump(self.y_test,open('./sift/y_test.pkl', 'wb'))
#             self.textEdit_2.setText(str(x_train))
#             self.textEdit_4.setText(str(x_test))
#             self.textEdit_3.setText(str(y_train))
#             self.textEdit_5.setText(str(y_test))
                         
        
    
    def egitim(self):
        self.algorıtma=self.comboBox.currentText()
        if self.algorıtma=="KNN":
            self.model1=KNeighborsClassifier(n_neighbors=5 , metric="minkowski")
            self.model1.fit(self.x_train,self.y_train)            
            y_pred= self.model1.predict(self.x_test)           
            self.textEdit.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, y_pred))))
            cm=confusion_matrix(self.y_test, y_pred)
            tp=cm[1,1]
            tn=cm[0,0]
            fp=cm[0,1]
            fn=cm[1,0]
            sen=tp/(tp+fn)
            spe=tn/(tn+fp)
            self.lineEdit.setText(str(sen))
            self.lineEdit_3.setText(str(spe))
            self.textEdit_6.setText(classification_report(self.y_test, y_pred))           
            plot_confusion_matrix( self.model1, self.x_test, self.y_test) 
            plt.savefig("./confusion/cm.png")
            self.pixmap = QPixmap("./confusion/cm.png")
            self.label_8.setPixmap(self.pixmap)
            self.ROC()
            plt.savefig("./ROC2.png")
            self.pixmap = QPixmap("./ROC2.png")
            self.label_17.setPixmap(self.pixmap)
            
        if self.algorıtma=="Logistic Regression":
            self.model1=LogisticRegression(random_state=0)
            self.model1.fit(self.x_train,self.y_train)
            self.y_pred = self.model1.predict(self.x_test) 
            self.textEdit.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, self.y_pred))))
            cm=confusion_matrix(self.y_test, self.y_pred)
            tp=cm[1,1]
            tn=cm[0,0]
            fp=cm[0,1]
            fn=cm[1,0]
            sen=tp/(tp+fn)
            spe=tn/(tn+fp)
            self.lineEdit_6.setText(str(sen))
            self.lineEdit_7.setText(str(spe))
            self.textEdit_8.setText(classification_report(self.y_test, self.y_pred))
            plot_confusion_matrix(self.model1, self.x_test, self.y_test) 
            plt.savefig("./confusion/cm3.png")
            self.pixmap = QPixmap("./confusion/cm3.png")
            self.label_13.setPixmap(self.pixmap)
            self.ROC()
            plt.savefig("./ROC.png")
            self.pixmap = QPixmap("./ROC.png")
            self.label_14.setPixmap(self.pixmap)
            
        if self.algorıtma=="Random Forest":
            self.model1=RandomForestClassifier(n_estimators=10 , criterion="entropy")
            self.model1.fit(self.x_train, self.y_train)
            y_pred=self.model1.predict(self.x_test)
            self.textEdit.setText(str("Karışıklık Matrisi:\n{}".format(confusion_matrix(self.y_test, y_pred))))
            cm=confusion_matrix(self.y_test, y_pred)
            tp=cm[1,1]
            tn=cm[0,0]
            fp=cm[0,1]
            fn=cm[1,0]
            sen=tp/(tp+fn)
            spe=tn/(tn+fp)
            self.lineEdit_4.setText(str(sen))
            self.lineEdit_5.setText(str(spe))
            self.textEdit_7.setText(classification_report(self.y_test, y_pred))           
            plot_confusion_matrix(self.model1, self.x_test, self.y_test) 
            plt.savefig("./confusion/cm2.png")
            self.pixmap = QPixmap("./confusion/cm2.png")
            self.label_12.setPixmap(self.pixmap)
            self.ROC()
            plt.savefig("./ROC1.png")
            self.pixmap = QPixmap("./ROC1.png")
            self.label_22.setPixmap(self.pixmap)
            
    def ROC(self):        
#        clf_reg = LogisticRegression(random_state=0);
#        clf_reg.fit(self.x_train, self.y_train);
#        y_score2 = clf_reg.predict_proba(self.x_test)
        pred_prob1 = self.model1.predict_proba(self.x_test)
        false_pozitive_rate = dict()
        true_pozitive_rate = dict()
        thresholds =dict()     
                 
        false_pozitive_rate, true_pozitive_rate, thresholds = roc_curve(self.y_test, pred_prob1[:,1], pos_label=1)   
        plt.figure(figsize=(4,3))                        
        plt.plot([0, 1], ls="--")        
        plt.plot(false_pozitive_rate, true_pozitive_rate, color='black', label='Daisy')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
      
        
    def yukle(self):
         self.dosyaadi, isim=QtWidgets.QFileDialog.getOpenFileName(None,"Görüntü Seç","","Veri Seti Türü(*.jpg)")
         self.label_15.setText(self.dosyaadi[55:])
         img=Image.open(self.dosyaadi)
         self.new_image=img.resize((250,250))
         self.new_image.save(self.dosyaadi)
         self.label_16.setStyleSheet("background-image : url('"+self.dosyaadi+"')")
         
    def CNN(self):
        
#        n_cols = self.x_train.shape[1]
#        self.X=self.X.reshape(224,224,3)
        self.degisken=float(self.comboBox_2.currentText())
        self.x_train, self.x_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,test_size=self.degisken, random_state=42)
        
        
        input_layer=tf.keras.Input([30,30,55])
        conv1=tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu') (input_layer)
        pool1=tf.keras.layers.MaxPooling2D(pool_size=(2,2)) (conv1)
        conv2=tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu')(pool1)
        pool2=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)) (conv2)
        conv3=tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu')(pool2)
        pool3=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)) (conv3)
        conv4=tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu')(pool3)
        pool4=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)) (conv4)
        flt=tf.keras.layers.Flatten()(pool4)
        dn1=tf.keras.layers.Dense(512,activation='relu')(flt)
        out=tf.keras.layers.Dense(512,activation='softmax')(dn1)
        
        model=tf.keras.Model(input_layer,out)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.summary()
        history = model.fit(self.x_train, 
                            self.y_train,
                            validation_data=(self.x_test, self.y_test),
                            batch_size=32, 
                            shuffle=True,
                            verbose=1,
                            epochs=20)
        y_pred=model.predict(self.x_test)
        print(y_pred)
        model.save('mymodel.h5')
        plt.figure(figsize=(4,3))
        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("./accu.png")
        self.pixmap = QPixmap("./accu.png")
        self.label_19.setPixmap(self.pixmap)
        plt.show()
        
        plt.figure(1)
        plt.figure(figsize=(4,3))

        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./loss.png")
        self.pixmap = QPixmap("./loss.png")
        self.label_20.setPixmap(self.pixmap)
        plt.show()
        
        false_pozitive_rate = dict()
        true_pozitive_rate = dict()
        thresholds =dict()     
        for i in range(5):    
            false_pozitive_rate[i], true_pozitive_rate[i], thresholds[i] = roc_curve(self.y_test, y_pred[:,i], pos_label=i)   
            
        plt.figure(figsize=(4,3))
             
        plt.plot([0, 1], ls="--")        
        plt.plot(false_pozitive_rate[0], true_pozitive_rate[0], color='black', label='Daisy')
        plt.plot(false_pozitive_rate[1], true_pozitive_rate[1], color='brown', label='Dandelion')
        plt.plot(false_pozitive_rate[2], true_pozitive_rate[2], color='pink', label='Rose')
        plt.plot(false_pozitive_rate[3], true_pozitive_rate[3], color='red', label='Sunflower')
        plt.plot(false_pozitive_rate[4], true_pozitive_rate[4], color='green', label='Tulip')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("./ROC.png")
        self.pixmap = QPixmap("./ROC.png")
        self.label_18.setPixmap(self.pixmap)
        plt.show()

        
    def tahminet(self):
       model = tf.keras.models.load_model(Path('./mymodel.h5'))
       classes=['daisy','dandelion','rose','sunflower','tulip']       
       new_img = load_img(self.dosyaadi, target_size=(224, 224))
       img = img_to_array(new_img)
       img = np.expand_dims(img, axis=0)
       prediction = model.predict(img)
       prediction = np.argmax(prediction,axis=1)
       self.lineEdit_2.setText(str(classes[np.argmax(prediction)]))

        
        

            
            
            