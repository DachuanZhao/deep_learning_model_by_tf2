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


print(tf.__version__)
APP_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(APP_DIR)

def plot_roc(fig_save_path:str,name_list:list,label_list_list:list,prediction_list_list:list)->None:
    """
    画出所有的
    """
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_one_roc(ax,name, label_list, prediction_list,**kwargs):
        """
        画出一个图
        """
        fp, tp, _ = sklearn.metrics.roc_curve(np.array(label_list),np.array(prediction_list))

        ax.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

    for i,(name,label_list,prediction_list) in enumerate(zip(name_list,label_list_list,prediction_list_list)):
        plot_one_roc(ax , name, label_list, prediction_list, color=color_list[i])

    ax.set_xlabel('False positives [%]')
    ax.set_ylabel('True positives [%]')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    fig.savefig(fig_save_path,quality=100)


def get_train_test_dataset_without_embeddings(train_date_csv_path:str,eval_data_csv_path:str,test_data_csv_path:str,word_count:int):
    """
    获得训练集合和测试集合,返回格式为:
    [x_train,y_train,x_eval,y_eval,x_test,y_test]
    """

    res = []
    word2index_dict = {"<PADDING>": 0, "<UNK>":1}
    delta_index = len(word2index_dict)

    #根据训练集挑选词汇
    word_count_dict = {}
    all_word_count = 0
    with open(train_date_csv_path) as f:
        for i,row in enumerate(f):
            row_x = row.strip().split("<SEP>")[0]
            for word in row_x.split(" "):
                all_word_count += 1
                if word in word_count_dict:
                    word_count_dict[word] += 1
                else:
                    word_count_dict[word] = 1

    filter_word_count = 0
    for i,temp_tuple in enumerate(sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)):
        if i > word_count - delta_index - 1:
            print("截断到词{}:{}".format(temp_tuple[0],temp_tuple[-1]))
            print("筛选出来的词频占总词频的{}".format(filter_word_count/all_word_count))
            break
        filter_word_count += temp_tuple[-1]
        word2index_dict[temp_tuple[0]] = i + delta_index

    for file_index,file_path in enumerate([train_date_csv_path,eval_data_csv_path,test_data_csv_path]):
        with open(file_path,newline="") as f:
            print("正在读取{}".format(file_path))
            x_list = []
            y_list = []

            for i,row in enumerate(f):
                if i%100000 == 0:
                    print("正在读取第{}行".format(i))
                row = row.strip()
                if len(row.split("<SEP>")) == 2:
                    row_y = int(row.split("<SEP>")[-1])
                else:
                    row_y = -1
                row_x = row.split("<SEP>")[0]
                row_x_list = row_x.split(" ")
                temp_list = []
                for word in row_x_list:
                    word = word.strip()
                    if not word in word2index_dict:
                        temp_list.append(word2index_dict["<UNK>"])
                    else:
                        temp_list.append(word2index_dict[word])
                x_list.append(temp_list)
                y_list.append(row_y)
            res.append(np.array(x_list))
            res.append(np.array(y_list))
            print("读取完成{}".format(file_path))
    return res,word2index_dict

def trans_gensim_word2vec2tf_embedding(word2vector_file_path:str):
    """
    把gensim的word2vec结果转化为tf.keras.layers.Embedding需要的结果
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

    return embeddings_matrix,word2vector_dict,word2index_dict

def get_train_test_dataset_with_embedding_matrix(train_date_csv_path:str,eval_data_csv_path:str,test_data_csv_path:str,word2index_dict:dict,delimiter = ",",quotechar = '"'):
    """
    获得训练集合和测试集合,返回格式为:
    [x_train,y_train,x_eval,y_eval,x_test,y_test]
    """

    res = []

    for file_path in [train_date_csv_path,eval_data_csv_path,test_data_csv_path]:
        with open(file_path,newline="") as f:
            #csv_reader = csv.reader(f,delimiter=delimiter,quotechar=quotechar)
            print("正在读取{}".format(file_path))
            x_list = []
            y_list = []

            #for i,row in enumerate(csv_reader):
            for i,row in enumerate(f):
                row = json.loads(row.strip())
                if i%100000 == 0:
                    print("正在读取第{}行".format(i))
                #row_y = int(row[-1].split("<SEP>")[-1])
                #row[-1] = row[-1].split("<SEP>")[0]
                #x_list.append([word2index_dict[word] if word in word2index_dict else 0 for word in row])
                #x_list.append([word2index_dict[str(word)] for word in row])
                #y_list.append(row_y)
                x_list.append([word2index_dict[str(word)] for word in row[:-1]])
                y_list.append(row[-1])
            res.append(np.array(x_list))
            res.append(np.array(y_list))
            print("读取完成{}".format(file_path))
    return res

def build_model(sentence_maxlen:int,word_count:int,embedding_dim:int,embeddings_matrix=None):
    """
    建立模型
    """

    inputs = tf.keras.Input(shape=(sentence_maxlen,),name="sentence_input")  # Returns an input placeholder
    if isinstance(embeddings_matrix,np.ndarray):
        hide_inputs = tf.keras.layers.Embedding(
            input_dim=word_count,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix),
            input_length=sentence_maxlen,
            trainable=False,
            name = "embeddings"
        )(inputs)
    else:
        hide_inputs = tf.keras.layers.Embedding(
            input_dim=word_count,
            output_dim=embedding_dim,
            input_length=sentence_maxlen,
            trainable=True,
            name = "embeddings"
        )(inputs)




    hide_inputs = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=embedding_dim,
            activation='tanh',
            dropout=0.1,
            recurrent_dropout=0.1,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            #参考你在训练RNN的时候有哪些特殊的trick？ - YJango的回答 - 知乎
            #https://www.zhihu.com/question/57828011/answer/155275958
        ),
        merge_mode='concat',
        name = "bi_lstm"
    )(hide_inputs)

    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid',use_bias=True,name="classify_dense")(hide_inputs)

    #outputs = tf.keras.layers.Dropout(0.1)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='bi_lstm_model')

    metric_list = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(threshold=0.5,name='accuracy'),
        tf.keras.metrics.Precision(thresholds=0.5,name='precision'),
        tf.keras.metrics.Recall(thresholds=0.5,name='recall'),
        tf.keras.metrics.AUC(num_thresholds=200,name='auc'),
    ]

    model.compile(
        loss = tf.keras.losses.binary_crossentropy,
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3,
            clipnorm=1.0,
            clipvalue=0.5,
        ),
        metrics = metric_list,
    )

    model.summary()

    tf.keras.utils.plot_model(model, './bi_lstm_model.png', show_shapes=True)

    return model

def build_model_callback():
    callback_file_path = "./model_callback/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=callback_file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./model_callback/logs",histogram_freq=1,update_freq='batch')

    return model_callback,tensorboard_callback

def train_data():

    sentence_maxlen = 5
    train_date_csv_path = r"/binary_classification/model_data/three_node_20191204/train.txt"
    eval_date_csv_path = r"/binary_classification/model_data/three_node_20191204/eval.txt"
    test_data_csv_path = r"/binary_classification/model_data/three_node_20191204/test.txt"

    model_path = r"./bi_lstm_model.h5"

    word2index_dict_path = "./word2index_dict.bin"
    word2vector_dict_path = "./word2vector_dict.bin"
    word2vector_file_path = r"/word2vector.bin"
    embeddings_matrix,word2vector_dict,word2index_dict = trans_gensim_word2vec2tf_embedding(word2vector_file_path)
    with open(word2vector_dict_path,"wb") as f:
        pickle.dump(word2vector_dict,f)
    word_count,embedding_dim = embeddings_matrix.shape

    #word_count = 10000
    #embedding_dim = 128
    #[x_train_nparray,y_train_nparray,x_eval_nparray,y_eval_nparray,x_test_nparray,y_test_nparray],word2index_dict = \
    #    get_train_test_dataset_without_embeddings(train_data_csv_path,eval_data_csv_path,test_data_csv_path,word_count)

    x_train_nparray,y_train_nparray,x_eval_nparray,y_eval_nparray,x_test_nparray,y_test_nparray = \
        get_train_test_dataset_with_embedding_matrix(train_date_csv_path,eval_date_csv_path,test_data_csv_path,delimiter = " ",word2index_dict=word2index_dict)

    with open(word2index_dict_path,"wb") as f:
        pickle.dump(word2index_dict,f)

    x_train_nparray = tf.keras.preprocessing.sequence.pad_sequences(x_train_nparray, sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
    x_eval_nparray = tf.keras.preprocessing.sequence.pad_sequences(x_eval_nparray, sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
    x_test_nparray = tf.keras.preprocessing.sequence.pad_sequences(x_test_nparray, sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
    print("len(word2index_idct) = {}".format(len(word2index_dict)))
    print("train_list[0]:{}:{}".format(x_train_nparray[0],y_train_nparray[0]))
    print("eval_list[0]:{}:{}".format(x_eval_nparray[0],y_eval_nparray[0]))
    print("test_list[0]:{}:{}".format(x_test_nparray[0],y_test_nparray[0]))

    model = build_model(sentence_maxlen,word_count,embedding_dim,embeddings_matrix)

    #model = build_model(sentence_maxlen,word_count,embedding_dim,embeddings_matrix=None)

    #'''
    model_callback = build_model_callback()

    history = model.fit(
        x=x_train_nparray,
        y=y_train_nparray,
        validation_data = (x_eval_nparray,y_eval_nparray),
        batch_size=256,
        epochs=50,
        verbose=1,
        callbacks=list(model_callback),
    )

    #保存模型
    model.save(model_path)

    #保存验证
    with open("./history_dict.bin","wb") as f_history:
        pickle.dump(history.history, f_history)
    #'''

    #model_path = r"./bi_lstm_model.h5"
    #model =  tf.keras.models.load_model(model_path)
    #model_path = r"./model_callback/weights.16-0.06.hdf5"
    #model.load_weights(model_path)
    #model.save(r"./{}.h5".format(".".join(os.path.basename(model_path).split(".")[:-1])))

    predict_batch_size = 2**17

    #验证模型,和compile里的相同
    test_evaluate_result = model.evaluate(
        x = x_test_nparray,
        y = y_test_nparray,
        batch_size = predict_batch_size,
        verbose = 0,
        workers = multiprocessing.cpu_count()*5,
        use_multiprocessing = True,
    )
    for key,value in zip(model.metrics_names,test_evaluate_result):
        print("{} : {}".format(key,value))

    eval_predict_probility_list = model.predict(
        x=x_eval_nparray,
        verbose=0,
        batch_size=predict_batch_size,
        workers=multiprocessing.cpu_count()*5,
        use_multiprocessing=True,
    )

    test_predict_probility_list = model.predict(
        x=x_test_nparray,
        verbose=0,
        batch_size=predict_batch_size,
        workers=multiprocessing.cpu_count()*5,
        use_multiprocessing=True,
    )

    with open("./model_test.txt","w") as f:
        for x_test,prob_list in zip(x_test_nparray,test_predict_probility_list):
            temp_list = [int(v) for v in list(x_test)]
            temp_list.append(float(prob_list[0]))
            f.write(json.dumps(temp_list) + "\n")

    #画roc
    plot_roc(fig_save_path = "./model_roc.png",
        name_list = ["eval","test"],
        label_list_list=[y_eval_nparray,y_test_nparray,],
        prediction_list_list=[
            [prob_list[0] for prob_list in eval_predict_probility_list],
            [prob_list[0] for prob_list in test_predict_probility_list],
        ]
    )

    real_probility_nparray = np.array(y_test_nparray,dtype=int)
    for interval in range(5,10,1):
        interval = interval / 10
        predict_probility_trans_nparray = np.fromiter((1 if prob_list[0]>interval else 0 for prob_list in test_predict_probility_list),dtype=int)
        assert len(predict_probility_trans_nparray) == len(real_probility_nparray)
        print("{} sklearn.metrics.classification_report:".format(interval))
        print(sklearn.metrics.classification_report(y_true=real_probility_nparray,y_pred=predict_probility_trans_nparray,digits=4))


def predict_model(input_sentence_file_path,output_sentence_file_path):

    sentence_maxlen = 5
    batch_size = 2**17

    model_path = r"./weights.06-0.17.h5"
    model =  tf.keras.models.load_model(model_path)

    word2index_dict_path = "./word2index_dict.bin"
    with open(word2index_dict_path,"rb") as f:
        word2index_dict = pickle.load(f)
    print({"<UNK>":word2index_dict["<UNK>"],"<PADDING>":word2index_dict["<PADDING>"]})

    trans_input_sentence_list = []
    input_sentence_list = []
    with open(input_sentence_file_path) as fi,open(output_sentence_file_path,"w") as fo:
        csv_reader = csv.reader(fi,dialect="excel")
        csv_writer = csv.writer(fo,dialect="excel")
        #每个batch都要被清空
        input_sentence_list = []
        trans_input_sentence_list = []
        for i,row in enumerate(csv_reader):
            input_sentence_list.append([v for v in row])
            if i % int(batch_size * multiprocessing.cpu_count()) == 0 and i != 0 :
                print("正在预测{}行".format(i))
                print("检查输入")
                print(input_sentence_list[0])
                trans_input_sentence_nparray = np.array([[word2index_dict[str(v)] for v in row] for row in input_sentence_list])
                print(trans_input_sentence_nparray[0])
                trans_input_sentence_nparray = tf.keras.preprocessing.sequence.pad_sequences(trans_input_sentence_nparray,
                    sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
                print("x_input的长度为{}".format(len(trans_input_sentence_nparray)))
                predict_probility_list = model.predict(
                    trans_input_sentence_nparray,
                    verbose=0,
                    batch_size=int(batch_size),
                    workers=multiprocessing.cpu_count()*5,
                    use_multiprocessing=True,
                )
                for i,(x,y) in enumerate(zip(input_sentence_list,predict_probility_list)):
                    if i % int(batch_size) == 0:
                        print("正在写入该batch第{}行".format(i))
                    x.extend(y)
                    csv_writer.writerow(x)
                #每个batch都要被清空
                input_sentence_list = []
        if input_sentence_list:
            print("检查输入")
            print(input_sentence_list[0])
            trans_input_sentence_nparray = np.array([[word2index_dict[str(v)] for v in row] for row in input_sentence_list])
            print(trans_input_sentence_nparray[0])
            trans_input_sentence_nparray = tf.keras.preprocessing.sequence.pad_sequences(trans_input_sentence_nparray,
                sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
            print(len(trans_input_sentence_nparray))
            predict_probility_list = model.predict(
                trans_input_sentence_nparray,
                verbose=0,
                batch_size=int(batch_size),
                workers=multiprocessing.cpu_count()*5,
                use_multiprocessing=True,
            )
            for i,(x,y) in enumerate(zip(input_sentence_list,predict_probility_list)):
                if i % int(batch_size) == 0:
                    print("x_input的长度为{}".format(len(trans_input_sentence_nparray)))
                x.extend(y)
                csv_writer.writerow(x)
            input_sentence_list = []

def predict_one():
    sentence_maxlen = 5
    model_path = r"./bi_lstm_model.h5"
    model =  tf.keras.models.load_model(model_path)

    word2index_dict_path = "./word2index_dict.bin"
    with open(word2index_dict_path,"rb") as f:
        word2index_dict = pickle.load(f)
    #print(word2index_dict)

    trans_input_sentence_nparray = np.array(([2374,62,47072,161,4051],))
    print(trans_input_sentence_nparray)
    trans_input_sentence_nparray = tf.keras.preprocessing.sequence.pad_sequences(trans_input_sentence_nparray,
        sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
    #predict_probility = model.predict(x=trans_input_sentence,verbose=1)

    print("开始预测")
    print(trans_input_sentence_nparray)

    assert trans_input_sentence_nparray.any()

    predict_probility_list = model.predict(
        x=trans_input_sentence_nparray,
        verbose=1,
        batch_size=128,
        workers=multiprocessing.cpu_count(),
        use_multiprocessing=True,
    )

    print(predict_probility_list)

if __name__ == "__main__":
    train_data()
    #predict_model(os.path.join(APP_DIR,"predicts/predict_three_node.csv"),"./three_node.csv")
    #predict_one()
