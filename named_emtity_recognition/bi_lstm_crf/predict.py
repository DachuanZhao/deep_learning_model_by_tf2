import tensorflow as tf
import tensorflow_addons as tf_ad
import numpy as np
import os
import json
import pickle
from utils import trans2index
from model import BiLstmCrfModel

APP_DIR  = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(APP_DIR,"data/word2index.bin"),"rb") as f:
    word2index_dict = pickle.load(f)

with open(os.path.join(APP_DIR,"data/tag2index.bin"),"rb") as f:
    tag2index_dict = pickle.load(f)
    index2tag_dict = {value:key for key,value in tag2index_dict.items()}

def predict_one_file(fi,fo,predict_batch_size:int):
    """预测一个文件"""
    x_word = []
    x_index = []
    for i,row in enumerate(fi):
        row_list = json.loads(row.strip())
        x_word.append(row_list)
        x_index.append([trans2index(word2index_dict,v) for v in row_list])
    tf.print("开始padding")
    x_npa = np.array(x_index)
    tf.print(x_npa)
    x_npa = tf.keras.preprocessing.sequence.pad_sequences(x_npa,dtype=np.int32,truncating="post", padding='post',value=0)
    del x_index

    logit_list, text_length_list = model.predict(x_npa,batch_size=predict_batch_size)

    tf.print("开始预测")
    y_tag = []
    for logit, text_length in zip(logit_list, text_length_list):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_length], model.transition_params)
        y_tag.append([trans2index(index2tag_dict,v) for v in viterbi_path])
    for x,y in zip(x_word,y_tag):
        fo.write(json.dumps({"word_list":x,"tag_list":y}) + "\n")


if __name__ == "__main__":
    #embedding_dim，要和train时一样
    embedding_dim = 128

    #lstm输出维度，要和train的时候一样
    lstm_unit_num = embedding_dim

    optimizer = tf.keras.optimizers.Adam(0.001)
    model = BiLstmCrfModel(lstm_unit_num=128,vocab_size=len(word2index_dict),tag_size=len(tag2index_dict),embedding_dim=128)
    ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
    ckpt.restore(tf.train.latest_checkpoint(os.path.join(APP_DIR,"model_callback/")))

    predict_path = os.path.join(APP_DIR,"data/ner_predict.txt")
    predict_output_path = os.path.join(APP_DIR,"data/ner_predict_output.txt")
    predict_batch_size = 128
    with open(predict_output_path,"w") as fo:
        with open(predict_path) as fi:
            predict_one_file(fi,fo,predict_batch_size)

