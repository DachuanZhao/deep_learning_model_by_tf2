# -*- coding: utf-8 -*-
import json
import nltk
import logging
import random
import copy
import pandas as pd
import os,sys,csv
import numpy as np
import datetime

logging.basicConfig(level=logging.INFO)
APP_DIR  = os.path.dirname(os.path.realpath(__file__))

logging.info(APP_DIR)

#output
output_file_folder = os.path.join(APP_DIR,"data")

#文件标识,注意每次修改这里，防止覆盖
flag = datetime.datetime.now().strftime("_%Y%m%d%H%M")

#ner识别之后的文件夹
ner_file_folder_path = r"./data/test/"
#ner_file_folder_path = os.path.join(APP_DIR,"test")

#word2vec文件
output_word2vec_file = os.path.join(output_file_folder,"word2vec_input.txt")

#ner文件
output_ner_file = os.path.join(output_file_folder,"ner_input.txt")

#ner输入文件
ner_file_path_list = []
for root, _dir, file_name_list in os.walk(ner_file_folder_path):
    for file_name in file_name_list:
        ner_file_path_list.append(os.path.join(root,file_name))


def replace_sentence_word_tokenize_list(sentence_word_tokenize_list,inside_ner_list):
    """
    替换词
    """
    delta = 0
    for inside_ner_obj in inside_ner_list:
        start = inside_ner_obj["start"] - delta
        end = inside_ner_obj["end"] - delta
        del sentence_word_tokenize_list[start:end]
        match_db_word_name = inside_ner_obj["match_db_word_name"]
        sentence_word_tokenize_list.insert(start,match_db_word_name)
        delta = delta + end - start - 1
    return sentence_word_tokenize_list



def main():
    with open(output_word2vec_file,"w+") as f_word2vec,open(output_ner_file,"w+") as f_ner:
        for i,file_path in enumerate(ner_file_path_list):
            with open(file_path) as f:
                for j,row in enumerate(f):
                    if j % 10000 == 0:
                        logging.info("{}_{}_{}".format(i,file_path,j))
                    temp_dict = json.loads(row.strip())
                    ner_list = temp_dict["ner_list"]
                    for ner_obj in ner_list:
                        sentence_word_tokenize_list = ner_obj["sentence_word_tokenize_list"]
                        inside_ner_list = ner_obj["ner_list"]
                        tag_list = ["O" for _ in sentence_word_tokenize_list]
                        for ner in inside_ner_list:
                            category = ner["db_category"].upper()
                            start = ner["start"]
                            end = ner["end"]
                            tag_list[start] = "B-{}".format(category)
                            for k in range(start + 1,end):
                                try:
                                    tag_list[k] = "I-{}".format(category)
                                except:
                                    logging.info(temp_dict["pubmed_id"])
                        for s,t in zip(sentence_word_tokenize_list,tag_list):
                            f_ner.write("{} {}\n".format(s,t))
                        f_ner.write("\n")
                        f_word2vec.write(json.dumps(sentence_word_tokenize_list) + "\n")

if __name__ == '__main__':
    main()

