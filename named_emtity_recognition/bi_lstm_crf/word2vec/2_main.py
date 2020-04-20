import gensim
import os
import multiprocessing
import logging
import sys
import json
import pandas as pd
logging.basicConfig(level=logging.INFO)

from gensim.test.utils import datapath

APP_DIR  = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

logging.info(APP_DIR)

class CustomLineSentece():

    def __init__(self,file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as fin:
            for row in fin:
                yield json.loads(row.strip())


def main():
    #file_path = os.path.join(APP_DIR,"data/word2vec_input.txt.test")
    file_path = os.path.join(APP_DIR,"data/word2vec_input.txt")

    sentences = CustomLineSentece(datapath(file_path))

    #for i,v in enumerate(sentences):
    #    logging.info("{}_{}".format(i,v))
    #    if i==10: assert 0

    model = gensim.models.word2vec.Word2Vec(
        sentences=sentences,
        size = 128,
        window = 5,
        min_count = 2,
        workers=multiprocessing.cpu_count(),
        iter=10,
        batch_words=2**13,
        )
    model.save("./word2vector.bin")
    model.wv.save_word2vec_format(
        fname="./vectors.txt",
        fvocab="./vocabulary.txt",
        binary=False)

if __name__ == "__main__":
    main()

