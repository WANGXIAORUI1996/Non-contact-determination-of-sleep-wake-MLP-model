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
import keras.backend as K
from itertools import count
import keras_metrics as km
from keras.callbacks import Callback
import tensorflow_addons as tfa
from sklearn.utils import class_weight



print("[INFO] 构建定义模型参数")
#opt = optimizers.RMSprop(learning_rate=0.00001, rho=0.9)
#num_epoch = 200

opt = tf.keras.optimizers.Adam(
    learning_rate=0.00001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    decay=0.00001,
    amsgrad=False )
num_epoch = 200

print("[INFO] 定义损失函数")
def focal_loss(gamma=1 , alpha=.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


print("[INFO] 输入训练集")
data = pd.read_excel('./SLEEPE/TRAINSET.xlsx')
#删除缺失值
data = data.dropna()
# 选择X和y
Y = data.iloc[:,9].values
x = data.iloc[:, 10:].astype('float').values
# 数据标准化
x_mean = x.mean(0)
x_std = x.std(0)
X = (x-x_mean)/x_std   
# 数据归一化     
print("[INFO] 构建K折交叉验证模型")
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
prs = []
aucs = []
mean_recall = np.linspace(0, 1, 100)
i = 0
cvscores = []  

for train, test in kfold.split(X, Y):

    model = models.Sequential()
    model.add(layers.Dense(400, activation='relu',kernel_regularizer=l2(0.0003)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(200, activation='relu',kernel_regularizer=l2(0.0003)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(100, activation='relu',kernel_regularizer=l2(0.0003)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(40, activation='relu',kernel_regularizer=l2(0.0003)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='relu',kernel_regularizer=l2(0.0003)))
    model.add(layers.Dense(1, activation='sigmoid',kernel_regularizer=l2(0.0003)))
    model.compile(optimizer=opt,
                  loss=[focal_loss(gamma=1 , alpha=.75)],
                  metrics=['accuracy',"TruePositives","TrueNegatives","FalsePositives","FalseNegatives",tfa.metrics.F1Score(name='F1-score',  average=None,num_classes=1,threshold=0.5)])
    history=model.fit(X[train], Y[train],validation_data=(X[test],Y[test]),epochs=num_epoch, batch_size=64, verbose=0)
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("[INFO]正在绘制结果曲线...")
    #plt.style.use("ggplot")
    plt.style.use("seaborn-ticks")
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    #plt.plot(history.history["val_F1-score"], label="val_F1-score")
    plt.title("loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("./SLEEPE/loss",dpi=500)
    plt.show() 
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))    
    cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print(model.metrics_names[2],scores[2])  
    print(model.metrics_names[3],scores[3])
    print(model.metrics_names[4],scores[4])
    print(model.metrics_names[5],scores[5])
    print("TPR",scores[2]/(scores[2]+scores[5]))
    print("TNR",scores[3]/(scores[3]+scores[4]))
    print("Precision",scores[2]/(scores[2]+scores[4]))
    print("F1 score",2*scores[2]/(2*scores[2]+scores[4]+scores[5]))
    # Compute PR curve and area the curve
    y_pred = model.predict(X[test]).ravel() 
    precision, recall, thresholds = precision_recall_curve(Y[test], y_pred)
    prs.append(interp(mean_recall, precision, recall))
    pr_auc = auc(recall, precision)
    aucs.append(pr_auc)
    plt.plot(recall, precision, lw=1, alpha=0.4, label='PR Fold %d (AUC = %0.2f)' % (i+1, pr_auc))
    i += 1
plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
mean_precision = np.mean(prs, axis=0)
mean_auc = auc(mean_recall, mean_precision)
std_auc = np.std(aucs)
plt.plot(mean_precision, mean_recall, color='b',label=r'Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall' )
plt.ylabel('Precision')
plt.title('PR curves')
plt.tick_params(axis='both')
plt.legend( prop={'size':13} , loc = 0)
plt.savefig("./SLEEPE/PRcurve",dpi=500)
plt.show()    

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
df.to_excel('./SLEEPE/TEST RESULT.xlsx',index=False)



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
df.to_excel('./SLEEPE/TRAIN RESULT.xlsx',index=False)


