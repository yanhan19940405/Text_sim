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
from gensim.models import Word2Vec
from random import shuffle
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


def handle_csv(path):
    data=[]
    file = open(path, encoding="utf-8")
    line = file.readline()
    while line:
        data.append(line.replace("\n", "").replace("\ufeff",''))
        line = file.readline()
    file.close()
    data_x = []
    for i in data:
        data_x.append(i.split("\t"))
    shuffle(data_x)
    Query1 = []
    Query2 = []
    label = []
    for a in data_x:
        Query1.append(a[0])
        Query2.append(a[1])
        label.append(a[2])
    return Query1,Query2,label

if __name__ == "__main__":
    Query1, Query2, label=handle_csv("data/train_data.txt")
    # Query1_1,Query2_2,label1=handle_csv("atec_nlp_sim_train_add.csv")
    # Query_a=Query1+Query1_1
    # Query_b=Query2+Query2_2
    # Queru_label=label+label1
    Query=Query1+Query2
    Q=[]
    for i in Query:
        Q.append(list(i))
    model = Word2Vec(Q, size=300, window=2,min_count=1)
    model.save("./model/Q_w2v_char.model")
    print("词向量模型训练完毕")
    print(1)
