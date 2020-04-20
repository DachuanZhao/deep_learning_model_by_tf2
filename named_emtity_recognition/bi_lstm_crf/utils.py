#code:utf8
import gensim
import numpy as np
import csv,os,sys
import tensorflow as tf
import pickle
import json
import multiprocessing
import datetime
import sklearn.metrics
import matplotlib.pyplot as plt

APP_DIR  = os.path.dirname(os.path.realpath(__file__))


def trans_gensim_word2vec2tf_embedding(word2vector_file_path:str):
    """把gensim的word2vec结果转化为tf.keras.layers.Embedding需要的结果
    """

    word2vec_model = gensim.models.Word2Vec.load(word2vector_file_path)

    #所有的词
    word_list = [word for word, word_info in word2vec_model.wv.vocab.items()]

    #词到index的映射
    word2index_dict = {"<PADDING>": 0, "<UNK>":1}

    #保存特殊词的padding
    specical_word_count = len(word2index_dict)

    #词到词向量的映射
    word2vector_dict = {}

    #初始化embeddings_matrix

    embeddings_matrix = np.zeros((len(word_list) + specical_word_count, word2vec_model.vector_size))
    #初始化unk为-1,1分布
    embeddings_matrix[word2index_dict["<UNK>"]] = (1 / np.sqrt(len(word_list) + specical_word_count) * (2 * np.random.rand(word2vec_model.vector_size) - 1))

    for i,word in enumerate(word_list):
        #从0开始
        word_index = i + specical_word_count
        word2index_dict[str(word)] = word_index
        word2vector_dict[str(word)] = word2vec_model.wv[word] # 词语：词向量
        embeddings_matrix[word_index] = word2vec_model.wv[word]  # 词向量矩阵

    #写入文件
    with open(os.path.join(APP_DIR,"data/word2index.bin"),"wb") as f:
        pickle.dump(word2index_dict,f)

    return embeddings_matrix,word2vector_dict,word2index_dict


def trans2index(word2index_dict,word):
    """转换"""
    if word in word2index_dict:
        return word2index_dict[word]
    else:
        if "<UNK>" in word2index_dict:
            return word2index_dict["<UNK>"]
        else:
            raise ValueError("没有这个值，请检查")


def trans_data2tf_data(data_file_path:str,word2index_dict=None):
    """把data文件转化为tf.data
    """
    tag2index_dict = {"<PADDING>":0}
    tag_index_count = len(tag2index_dict)
    x_list = []
    y_list = []
    #读取数据，并转化为index
    with open(data_file_path,encoding="utf-8") as f:
        if word2index_dict is None:
            #todo
            pass
        else:
            temp_x_list = []
            temp_y_list = []
            for i,row in enumerate(f):
                if i % 10**7 == 0:
                    tf.print("已经读取{}行".format(i))
                row = row.strip()
                if row:
                    x,y = row.split(" ")
                    if not y in tag2index_dict:
                        tag2index_dict[y] = tag_index_count
                        tag_index_count += 1
                    temp_x_list.append(trans2index(word2index_dict,x))
                    temp_y_list.append(tag2index_dict[y])
                else:
                    if temp_x_list and temp_y_list:
                        x_list.append(temp_x_list)
                        y_list.append(temp_y_list)
                    else:
                        tf.print(temp_x_list,temp_x_list)
                    temp_x_list = []
                    temp_y_list = []
    tf.print("x_list[:3]:{}".format(x_list[:3]))
    tf.print("y_list[:3]:{}".format(y_list[:3]))

    #写入文件
    with open(os.path.join(APP_DIR,"data/tag2index.bin"),"wb") as f:
        pickle.dump(tag2index_dict,f)

    #这里需要省略比较长的句子,取正态分布的99.7
    x_max_length0 = np.max(np.array([len(v) for v in x_list]))
    x_max_length = int(np.max(np.percentile(np.array([len(v) for v in x_list]),99.7)))
    y_max_length = int(np.max(np.percentile(np.array([len(v) for v in y_list]),99.7)))
    tf.print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))

    x_npa = np.array(x_list)
    del x_list
    y_npa = np.array(y_list)
    del y_list

    x_npa = tf.keras.preprocessing.sequence.pad_sequences(x_npa,maxlen=x_max_length,dtype=np.int32,truncating="post", padding='post',value=0)
    y_npa = tf.keras.preprocessing.sequence.pad_sequences(y_npa,maxlen=y_max_length,dtype=np.int32,truncating="post", padding='post',value=0)

    tf.print("x_npa[-3:]:{}".format(x_npa[-3:]))
    tf.print("y_npa[-3:]:{}".format(y_npa[-3:]))

    return tf.data.Dataset.from_tensor_slices((x_npa,y_npa)),tag2index_dict
