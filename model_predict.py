import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.corpora import LowCorpus
from gensim.corpora import Dictionary
from keras import optimizers
import re
from sklearn.metrics import classification_report
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers.merge import concatenate
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import pickle
from keras.utils import plot_model
import  tensorflow as tf
import os
from keras.models import load_model
from gensim.models import Word2Vec
from attentionlevel import Attention
def euclidean_distance(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))
    # K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
def pos_matrix(maxlen: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(maxlen)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

data=[]
path="./data/dev_set.csv"
file = open(path, encoding="utf-8")
line = file.readline()
while line:
    data.append(line.replace("\n", "").replace("\ufeff",''))
    line = file.readline()
file.close()
data_x = []
for i in data:
    data_x.append(i.split("\t"))

Q_a = []
Q_b = []
id=[]
label = []
for a in data_x:
    id.append(a[0])
    Q_a.append(a[1])
    Q_b.append(a[2])


pkl_file = open('./dic/vocab.pkl', 'rb')
data3 = pickle.load(pkl_file)
pkl_file.close()
Q_a_encoder=[]
Q_b_encoder=[]
for a in Q_a[1:]:
    Q_a_encoder.append(jieba.lcut(a))
for b in Q_b[1:]:
    Q_b_encoder.append(jieba.lcut(b))
for count in range(len(Q_a_encoder)):
    for row in range(len(Q_a_encoder[count])):
        if Q_a_encoder[count][row] in data3:
            Q_a_encoder[count][row] = data3[Q_a_encoder[count][row]]
        else:
            Q_a_encoder[count][row] = data3["UNK"]
for countb in range(len(Q_b_encoder)):
    for rowb in range(len(Q_b_encoder[countb])):
        if Q_b_encoder[countb][rowb] in data3:
            Q_b_encoder[countb][rowb] = data3[Q_b_encoder[countb][rowb]]
        else:
            Q_b_encoder[countb][rowb] = data3["UNK"]
Q_a_test = pad_sequences(Q_a_encoder, maxlen=60)
Q_b_test = pad_sequences(Q_b_encoder, maxlen=60)
filepath = "./model/sim.h5"
dict1={"pos_matrix":pos_matrix,'Attention': Attention}
model = load_model(filepath, compile=False,custom_objects=dict1)
y_pred=model.predict([Q_a_test,Q_b_test])
print("label",y_pred)
y_value=[]
dis=[]
for i in y_pred:
    for a in i:
        dis.append(a)
for k in range(len(dis)):
    if dis[k]>=0.5:
        dis[k]=1
    elif dis[k]<0.5:
        dis[k]=0
dataframe = pd.DataFrame({'qid':[int(m) for m in id[1:]],'label':[int(n) for n in dis]})
columns = ['qid','label']
dataframe.to_csv("./data/result_20.csv", index=False, sep='\t',header=None,columns=columns)
print("ok")
