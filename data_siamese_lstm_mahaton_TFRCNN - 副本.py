import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from gensim.models import word2vec
import gensim
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
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model
from attentionlevel import Attention
from sklearn.metrics.pairwise import cosine_similarity
from layernormalize import LayerNormalize
import  tensorflow as tf
from random import shuffle
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
def exponent_neg_manhattan_distance(sent_left, sent_right):
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))
def pos_matrix(maxlen: int, d_emb: int) -> np.array:#position embbedding
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(maxlen)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc

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

def weight_share_networks(input_shape):
    input = Input(shape=input_shape)
    # lstm1 = Bidirectional(LSTM(300, return_sequences=True))(input)
    # lstm1 = Dropout(0.5)(lstm1)
    cnn1 = Convolution1D(300, kernel_size=1, padding='same', dilation_rate=1,
                         activation='relu')(input)
    cnn1 = Convolution1D(300, kernel_size=3, padding='same', dilation_rate=2,
                         activation='relu')(cnn1)
    cnn1 = Convolution1D(300, kernel_size=5, padding='same', dilation_rate=5, activation='relu')(cnn1)
    cnn1 = Convolution1D(300, kernel_size=1, padding='same', dilation_rate=1,
                         activation='relu')(cnn1)
    cnn1 = Convolution1D(300, kernel_size=3, padding='same', dilation_rate=2,
                         activation='relu')(cnn1)
    cnn1 = Convolution1D(300, kernel_size=5, padding='same', dilation_rate=5, activation='relu')(cnn1)


    # lstm2 = Bidirectional(LSTM(50))(lstm1)
    # lstm2 = Dropout(0.5)(lstm2)

    # cnn1 = Convolution1D(300, kernel_size=3, padding='same', activation='relu')(lstm1)
    # for i in range(4):
    #     cnn1 = Convolution1D(300, kernel_size=2 * i + 1, padding='same', dilation_rate=1,
    #                          activation='relu')(cnn1)
    #     cnn1 = Convolution1D(300, kernel_size=2 * i + 1, padding='same', dilation_rate=2,
    #                          activation='relu')(cnn1)
    #     cnn1 = Convolution1D(300, kernel_size=2 * i + 1, padding='same', dilation_rate=5,
    #                          activation='relu')(cnn1)
    # cnn1 = Dense(units=K.int_shape(cnn1)[-1], activation='relu')(cnn1)

    # cnn1 = Convolution1D(300, kernel_size=1, padding='same', dilation_rate=1,
    #                      activation='relu')(input)
    # cnn1 = Convolution1D(300, kernel_size=3, padding='same', dilation_rate=2,
    #                      activation='relu')(cnn1)
    # cnn1 = Convolution1D(300, kernel_size=5, padding='same', dilation_rate=5,
    #                      activation='relu')(cnn1)


    # tim=Dense(15)(added_4)
    # print("tim",K.int_shape(tim))
    # L_1 = LayerNormalize(output_dim=K.int_shape(tim)[-1])(tim)
    # print("L1", K.int_shape(L_1))
    # cnn1 = Convolution1D(256, kernel_size=5, padding='same', strides=1, activation='relu')(embed)
    # cnn1 = MaxPool1D(pool_size=5, strides=1, padding='same')(cnn1)
    # BRN_BP = Dense(units=K.int_shape(L_1)[-1], activation='relu')(L_1)
    return Model(input, cnn1)


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


class ManDist(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)


    def build(self, input_shape):
        super(ManDist, self).build(input_shape)


    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        # self.result=cosine_similarity(x[0],x[1])
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
def contrastive_loss(y_true, y_pred):#对比损失函数
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)



def compute_accuracy(y_true, y_pred):#计算测试集准确率
    pred = y_pred.flatten() < 0.5
    print(pred)
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred): #模型训练过程匹配程度
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def sim(left,right):
    return K.abs(left - right)


class epoch_lr(keras.callbacks.Callback):
    def __init__(self):
        self.num_passed_batchs = 0#每一轮batch起点
        self.warmup_epochs = 2# warm_up机制轮次数

    def on_batch_begin(self, batch, logs=None):
        if self.params['steps'] == None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']

        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            # 前2个epoch中，学习率线性地从零增加到0.0001
            K.set_value(self.model.optimizer.lr,
                        0.00001 * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
            self.num_passed_batchs += 1

def creat_model(wordindex,d_model,matrix,maxlen,X_train, y_train,X_test, y_test):
    embedding_layer = Embedding(len(wordindex) + 1, d_model, weights=[matrix], input_length=maxlen,trainable=False)
    left = Input(shape=(maxlen,), dtype='float32', name="Q_a")
    right = Input(shape=(maxlen,), dtype='float32', name='Q_b')
    embedding_left=embedding_layer(left)
    pos_left = keras.layers.Embedding(len(wordindex) + 1, d_model, trainable=False, input_length=maxlen,
                                 name='PositionEmbedding_left',
                                 weights=[pos_matrix(maxlen=len(wordindex) + 1, d_emb=d_model)])(left)
    print(K.int_shape(embedding_left))
    print(K.int_shape(pos_left))
    added_left= keras.layers.Add()([embedding_left, pos_left])
    # added_left=Attention(h=1, output_dim=K.int_shape(added_left)[-1])(added_left)
    print("left", K.int_shape(added_left))
    embedding_right=embedding_layer(right)
    pos_right= keras.layers.Embedding(len(wordindex) + 1, d_model, trainable=False, input_length=maxlen,
                                      name='PositionEmbedding_right',
                                      weights=[pos_matrix(maxlen=len(wordindex) + 1, d_emb=d_model)])(right)
    added_right = keras.layers.Add()([embedding_right, pos_right])
    # added_right = Attention(h=1, output_dim=K.int_shape(added_right)[-1])(added_right)
    print("right", K.int_shape(added_right))
    shared_lstm = weight_share_networks(input_shape=(maxlen,d_model))
    left_output = shared_lstm(added_left)
    right_output = shared_lstm(added_right)
    # distance =  ManDist()([left_output,right_output])
    distance=Lambda(lambda ten:K.exp(-K.sum(K.abs(ten[0]-ten[1]),axis=-1,keepdims=True)))([left_output,right_output])
    flatten_layer2 = Flatten()(distance)
    sof=Dense(1, activation='sigmoid')(flatten_layer2)
    model = Model(inputs=[left,right], outputs=sof)
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    model.summary()
    # from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file="./image/model.png", show_shapes=True)

    earlystopping = EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=1, verbose=2, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.00001,
                                  patience=1, min_lr=0.00005, mode='auto')
    history = model.fit(X_train, y_train, verbose=1, batch_size=batch_size, epochs=n_epoch,
                        validation_data=(X_test, y_test),
                        callbacks=[reduce_lr,epoch_lr(),earlystopping])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./model/result_manhaton_acc.png")


    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./model/result_manhaton_loss.png")

    filepath = "./model/sim.h5"
    model.save(filepath=filepath, include_optimizer=True)
    y_pred = model.predict(X_test)
    te_acc = compute_accuracy(y_test, y_pred)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss', 'val_loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'acc', 'val_acc')
    plt.savefig("./model/result_manhaton.png")

    print(y_pred)
    y_value=[]
    for i in y_pred:
        for a in i:
           y_value.append(a)
    for k in range(len(y_value)):
        if y_value[k]>=0.5:
            y_value[k]=1
        elif y_value[k]<0.5:
            y_value[k]=0
    print(classification_report(y_test, y_value))
    return model,y_pred,y_test,y_value

if __name__ == "__main__":
    Query1, Query2, label=handle_csv("data/train_data.txt")
    # Query1_1,Query2_2,label1=handle_csv("atec_nlp_sim_train_add.csv")
    Query_a=Query1
    Query_b=Query2
    Query_label=label
    for m in range(len(Query_label)):
        if str(1) in Query_label[m]:
            Query_label[m]=1
        elif str(0) in Query_label[m]:
            Query_label[m]=0
    # y= keras.utils.to_categorical(Query_label, num_classes=2)
    y=Query_label
    Query = Query_a + Query_b
    Q = []
    seq_maxlen=[]
    for i in Query:
        seq_maxlen.append(len(i))
        Q.append(" ".join(jieba.cut(i)))
        # Q.append(" ".join(list(i)))
    # maxlen=max(seq_maxlen)
    maxlen=60
    token=Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    token.fit_on_texts(Q)
    sequences = token.texts_to_sequences(Q)
    wordindex = token.word_index
    data = pad_sequences(sequences, maxlen=maxlen)
    # labels = keras.utils.to_categorical([int(i) for i in Query_label], num_classes=2)
    for index in wordindex:
        wordindex[index] = wordindex[index] + 1
    wordindex['PAD'] = 0
    wordindex['UNK'] = 1
    output = open('./dic/vocab.pkl', 'wb')
    pickle.dump(wordindex, output)
    output.close()
    model = gensim.models.Word2Vec.load('./model/Q_w2v_new.model')
    embedding_matrix = np.zeros((len(wordindex) + 1, 300))
    for word, i2 in wordindex.items():
        if word in model:
            embedding_matrix[i2] = np.asarray(model[word])
        elif word not in model:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i2] = np.random.uniform(-0.25, 0.25, 300)
    #分别对双输入K-V对进行分词编码处理
    Q_a=[]
    Q_b=[]
    Q_a_encoder=[]
    Q_b_encoder=[]
    for a in Query_a:
        Q_a.append(jieba.lcut(a))
        # Q_a.append(list(a))
    for b in Query_b:
        Q_b.append(jieba.lcut(b))
        # Q_b.append(list(b))
    for count in range(len(Q_a)):
        for row in range(len(Q_a[count])):
            if Q_a[count][row] in wordindex:
                Q_a[count][row]=wordindex[Q_a[count][row]]
            else:
                Q_a[count][row] = wordindex["UNK"]
    for countb in range(len(Q_b)):
        for rowb in range(len(Q_b[countb])):
            if Q_b[countb][rowb] in wordindex:
                Q_b[countb][rowb]=wordindex[Q_b[countb][rowb]]
            else:
                Q_b[countb][rowb] = wordindex["UNK"]
    Q_a=pad_sequences(Q_a, maxlen=maxlen)
    Q_b=pad_sequences(Q_b, maxlen=maxlen)
    k=80000
    Q_a_train=Q_a[0:k]
    Q_b_train=Q_b[0:k]
    label_train=y[0:k]
    batch_size = 32
    n_epoch = 100
    testa=80000
    testb=100000
    model,y_pred,y_test,y_value=creat_model(wordindex=wordindex,d_model=300,matrix=embedding_matrix,maxlen=maxlen,X_train=[Q_a_train,Q_b_train], y_train=label_train,X_test=[Q_a[testa:testb],Q_b[testa:testb]], y_test=y[testa:testb])

    print(1)










