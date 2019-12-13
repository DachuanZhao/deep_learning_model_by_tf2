import gensim
import os
import multiprocessing
import logging
import sys
import json
import pandas as pd
import csv
logging.basicConfig(level=logging.INFO)

from gensim.test.utils import datapath

#参数，只需要修改这里,todo

class Word2VecModel():
    """word2vec使用的类"""

    def __init__(self,model_file_path):
        self.model_file_path = model_file_path
        self.model = gensim.models.word2vec.Word2Vec.load(model_file_path)

    def get_similarity(self,word=None):
        """计算相似度,返回同义词"""

        if not isinstance(word,str):
            word = str(word)
        #logging.info('input:{},similar:{}'.format(word,self.model.wv.most_similar(word)))
        return self.model.wv.most_similar(word)

class CustomLineSentece():
    """逐行读取训练文件的类，文件的格式为每行一个list,例：
    ["i","love","China"]
    ["you","love","USA"]
    """

    def __init__(self,file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as fin:
            for row in fin:
                yield json.loads(row.strip())

def main():
    """训练"""

    #语料文件
    file_path = os.path.join(APP_DIR,"data/output_file_random_walk_no_confidence_20191115_160649_word2vec.txt")
    sentences = CustomLineSentece(datapath(file_path))
    model = gensim.models.word2vec.Word2Vec(
        sentences=sentences,
        size = 128,
        window = 9,
        min_count = 1,
        workers=multiprocessing.cpu_count() * 5,
        iter=10,
        batch_words=8192,
        )
    model.save("./word2vector.bin")
    model.wv.save_word2vec_format(
        fname="./vectors.txt",
        fvocab="./vocabulary.txt",
        binary=False)


def calculate_similarity():
    """计算相似度"""

    word_list = [
        "insulin",
        "vitamin a",
        ]

    new_model = Word2VecModel("./word2vector.bin")
    old_model = Word2VecModel("./without_confidence/word2vector.bin")

    df = pd.DataFrame()

    for  word in word_list:

        word  = word.lower()

        if not word in name_id_dict:
            logging.info("{}不在字典中".format(word))
            continue

        temp_list = []

        for i,value in enumerate(new_model.get_similarity(word)):
            temp_word,similarity = value
            logging.info(temp_word)
            temp_list.append(temp_word)

        temp_list.append("")

        for i,value in enumerate(old_model.get_similarity(word)):
            temp_word,similarity = value
            logging.info(temp_word)
            temp_list.append(temp_word)

        logging.info(temp_list)
        logging.info(word)

        df[word]  = temp_list

    logging.info(df)
    df.to_excel("./similarity.xlsx",encoding="utf-8-sig")

def test():
    """测试导入"""
    word_vec = gensim.models.KeyedVectors.load("./word2vector.bin")
    return word_vec


#其他代码
APP_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(APP_DIR)
name_id_json_path = os.path.join(APP_DIR,'data/name2id.json')
with open(name_id_json_path) as f:
    name_id_dict  = json.load(f)
    id_name_dict = {value:key for key,value in name_id_dict.items()}

if __name__ == "__main__":
    main()
    #calculate_similarity()
    #test()
