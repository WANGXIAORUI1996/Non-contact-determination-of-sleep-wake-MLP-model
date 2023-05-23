import keras
import h5py
import matplotlib
import tensorflow as tf
from tensorflow import optimizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Dense, activation
from tensorflow.keras.optimizers import SGD
from keras import initializers 
from keras import activations
from keras import regularizers
from my_utils import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from _pickle import dump
import pandas as pd
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
from sklearn.metrics import f1_score,roc_curve,auc,RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn import svm, datasets
from scipy import interp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score



print("[INFO] 构建定义模型参数")
opt = optimizers.RMSprop(learning_rate=0.0001, rho=0.9)
num_epoch = 50





print("[INFO] 输入训练集")
data = pd.read_excel('./SLEEPE/TRAINSET.xlsx')
#删除缺失值
data = data.dropna()
# 选择X和y
Y = data.iloc[:,9].values
# 为了训练方便，暂时把2改为1，后续预测结果还是0和2
#Y = np.where(y == 2, 1, 0)
x = data.iloc[:, 10:].astype('float').values
# 数据标准化
x_mean = x.mean(0)
x_std = x.std(0)
X = (x-x_mean)/x_std        




model = models.Sequential()
model.add(layers.Dense(100, activation='relu',kernel_regularizer=l2(0.0003)))
model.add(layers.Dense(40, activation='relu',kernel_regularizer=l2(0.0003)))
model.add(layers.Dense(10, activation='relu',kernel_regularizer=l2(0.0003)))
model.add(layers.Dense(1, activation='sigmoid',kernel_regularizer=l2(0.0003)))
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',"TruePositives","TrueNegatives","FalsePositives","FalseNegatives"])
history=model.fit(X, Y,epochs=num_epoch, batch_size=16, verbose=0)


print("[INFO] 输出保存模型")
model.save('./SLEEPE/model.h5')


print("[INFO] 测试集结果")
test_data=pd.read_excel('./SLEEPE/TESTSET.xlsx')
x0=test_data.iloc[:, 10:].values
#标准化
x0=(x0-x_mean)/x_std
#预测
y0=model.predict(x0).reshape(-1)
y0=np.where(y0<0.5,0,1)
#把结果保存到表里面
df=test_data.copy()
df.insert(0, 'PREDICT',y0 )
#保存表
df.to_excel('./SLEEPE/TESTRESULT.xlsx',index=False)



print("[INFO] 训练集结果")
test_data1=pd.read_excel('./SLEEPE/TRAINSET.xlsx')
x1=test_data1.iloc[:, 10:].values
#标准化
x1=(x1-x_mean)/x_std
#预测
y1=model.predict(x1).reshape(-1)
y1=np.where(y1<0.5,0,1)
#把结果保存到表里面
df=test_data1.copy()
df.insert(0, 'PREDICT',y1 )
#保存表
df.to_excel('./SLEEPE/TRAINRESULT.xlsx',index=False)
