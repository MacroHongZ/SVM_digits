# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:45:26 2021

@author: WHZ
"""
from sklearn import svm
from sklearn.datasets  import load_digits
from sklearn.model_selection  import train_test_split
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 

# 加载数据
mnist = load_digits()

# 部分数据展示
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    # 初始化子图:在8×8的网格中，在第i+1个位置添加一个子图
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # 在第i个位置显示图像
    ax.imshow(mnist.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # 用目标值标记图像 bbox=dict(facecolor='red', alpha=0.5)
    ax.text(0, 7, str(mnist.target[i]))
plt.show()

# 数据集划分
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.20,random_state=40)

# 基于t_SNE的数据可视化
X_std = StandardScaler().fit_transform(x) 
tsne = TSNE(n_components=2) 
X_tsne = tsne.fit_transform(X_std) 
X_tsne_data = np.vstack((X_tsne.T, y)).T 
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
df_tsne[['class']] = df_tsne[['class']].astype('category')

plt.figure(figsize=(8, 8)) 
sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2') 
plt.show()

# 基于PCA的数据可视化
X_pca = PCA(n_components=2).fit_transform(X_std) 
X_pca_data = np.vstack((X_pca.T, y)).T
df_pca = pd.DataFrame(X_pca_data, columns=['Dim1', 'Dim2', 'class'])
df_pca[['class']] = df_pca[['class']].astype('category')

plt.figure(figsize=(8, 8)) 
sns.scatterplot(data=df_pca, hue='class', x='Dim1', y='Dim2') 
plt.show()

# 定义线性 SVM 模型
# model = svm.LinearSVC(C=500, random_state=49)

# 非线性 SVM 模型，核函数可选：linear、poly、rbf、sigmoid
model = svm.NuSVC(kernel='rbf', random_state=49)
# 模型训练
model.fit(x, y)

# 模型预测
pre_y = model.predict(test_x)
# 计算评估指标
print(classification_report(test_y, pre_y))

# 预测结果可视化
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(test_x[i].reshape((8,8)), cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(test_y[i]))
    ax.text(7, 7, str(pre_y[i]), bbox=dict(facecolor='red', alpha=0.5))
plt.show()



