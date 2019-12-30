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
import time

tf.print(tf.__version__)
APP_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(APP_DIR)

#模型编写
def get_angles(pos, i, embedding_dim):
    """
    计算位置编码
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embedding_dim))
    return pos * angle_rates


def positional_encoding(position, embedding_dim):
    """
    获得位置编码
    """

    #angle_rads.shape = [position , embedding_dim]
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(embedding_dim)[np.newaxis, :],
                            embedding_dim)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq,padding_token=0):
    """
    遮挡一批序列中所有的填充标记（pad tokens）。这确保了模型不会将填充作为输入。
    """
    seq = tf.dtypes.cast(tf.math.equal(seq, padding_token), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
        输出，注意力权重
    """

    matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.dtypes.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    多头注意力
    """
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        assert embedding_dim % self.num_heads == 0

        self.depth = embedding_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(embedding_dim)
        self.wk = tf.keras.layers.Dense(embedding_dim)
        self.wv = tf.keras.layers.Dense(embedding_dim)

        self.dense = tf.keras.layers.Dense(embedding_dim)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, embedding_dim)
        k = self.wk(k)  # (batch_size, seq_len, embedding_dim)
        v = self.wv(v)  # (batch_size, seq_len, embedding_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.embedding_dim))  # (batch_size, seq_len_q, embedding_dim)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, embedding_dim)

        return output, attention_weights


def point_wise_feed_forward_network(embedding_dim, dff):
    """
    点式前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数。
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, embedding_dim)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """
    编码器层
    """
    def __init__(self, embedding_dim, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_dim, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, is_dropout_training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embedding_dim)
        attn_output = self.dropout1(attn_output, training=is_dropout_training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embedding_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.dropout2(ffn_output, training=is_dropout_training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embedding_dim)

        return out2


class Encoder(tf.keras.layers.Layer):
    """
    编码机器
    """
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                sentence_maxlen, dropout_rate=0.1 ,embeddings_matrix = None , is_embeddings_training=False):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.sentence_maxlen = sentence_maxlen

        embedding_param_dict = {
            "input_dim" : input_vocab_size,
            "output_dim" : embedding_dim,
            "trainable" : is_embeddings_training,
            "input_length" : sentence_maxlen
        }
        if not (embeddings_matrix is None):
            tf.print("导入了预训练的embedding_matrix")
            embedding_param_dict["embeddings_initializer"] = tf.keras.initializers.Constant(embeddings_matrix)
        else:
            embedding_param_dict["trainable"] = True
        if embedding_param_dict["trainable"] == True:
            tf.print("embedding_matrix加入了fine-tune")
        else:
            tf.print("embedding_matrix未加入fine-tune")

        self.embedding = tf.keras.layers.Embedding(**embedding_param_dict)

        #提前计算的位置embedding，多加上两个标志位置备用
        maximum_position_encoding = sentence_maxlen + 2
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.embedding_dim)


        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, dff, dropout_rate)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, is_dropout_training, mask):

        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=is_dropout_training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, is_dropout_training, mask)

        return x  # (batch_size, input_seq_len, embedding_dim)



class ClassifyTransformer(tf.keras.Model):
    """二分类transformer模型
    """

    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                dropout_rate, sentence_maxlen , embeddings_matrix = None , is_embeddings_training=False):
        """参数 :
        num_layers : encoder_layer的个数
        embedding_dim : embedding的层数

        """
        super(ClassifyTransformer, self).__init__()

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff,
                            input_vocab_size, sentence_maxlen , dropout_rate,
                            embeddings_matrix, is_embeddings_training)

        #最后二分类，0-1
        self.final_layer = tf.keras.layers.Dense(units=1, activation='sigmoid',use_bias=True,name="classify_dense")

    def call(self, inp, is_dropout_training=False, enc_padding_mask=None):

        if enc_padding_mask is None:
            enc_padding_mask = create_padding_mask(inp)

        enc_output = self.encoder(inp, is_dropout_training, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_dim)

        #把每句话拉长为一句话
        enc_output = tf.reshape(enc_output,[-1, self.encoder.sentence_maxlen * self.encoder.embedding_dim])

        final_output = self.final_layer(enc_output)

        return final_output



class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """"""
    def __init__(self, embedding_dim, warmup_steps=4000):
        super(CustomLearningRateSchedule, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding_dim = tf.cast(self.embedding_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(arg1, arg2)



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


def get_train_test_dataset_without_embeddings(train_data_csv_path:str,eval_data_csv_path:str,test_data_csv_path:str,word_count:int):
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
    with open(train_data_csv_path) as f:
        for i,row in enumerate(f):
            row_list = json.loads(row.strip())
            row_x = row_list[:-1]
            for word in row_x:
                all_word_count += 1
                if word in word_count_dict:
                    word_count_dict[word] += 1
                else:
                    word_count_dict[word] = 1
    tf.print("总共有词: {}个".format(len(word_count_dict)))

    filter_word_count = 0
    for i,temp_tuple in enumerate(sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)):
        if i > word_count - delta_index - 1:
            tf.print("截断到词{}:{}".format(temp_tuple[0],temp_tuple[-1]))
            tf.print("筛选出来的词频占总词频的{}".format(filter_word_count/all_word_count))
            break
        filter_word_count += temp_tuple[-1]
        word2index_dict[temp_tuple[0]] = i + delta_index

    #生成训练集，验证集，测试集
    for file_index,file_path in enumerate([train_data_csv_path,eval_data_csv_path,test_data_csv_path]):
        with open(file_path,newline="") as f:
            tf.print("正在读取{}".format(file_path))
            x_list = []
            y_list = []

            for i,row in enumerate(f):
                if i%100000 == 0:
                    tf.print("正在读取第{}行".format(i))
                row_list = json.loads(row.strip())
                row_x_list = row_list[:-1]
                row_y = row_list[-1]
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
            tf.print("读取完成{}".format(file_path))
    return res,word2index_dict

def trans_gensim_word2vec2tf_embedding(word2vector_file_path:str):
    """
    把gensim的word2vec结果转化为tf.keras.layers.Embedding需要的结果
    """

    #word2vector_file_path = "./word2vector.bin"
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

    tf.print("embeddings_matrix = ")
    tf.print(embeddings_matrix)
    return embeddings_matrix,word2vector_dict,word2index_dict

def get_train_test_dataset_with_embedding_matrix(train_data_csv_path:str,eval_data_csv_path:str,test_data_csv_path:str,word2index_dict:dict,delimiter = ",",quotechar = '"'):
    """
    获得训练集合和测试集合,返回格式为:
    [x_train,y_train,x_eval,y_eval,x_test,y_test]
    """

    res = []

    for file_path in [train_data_csv_path,eval_data_csv_path,test_data_csv_path]:
        with open(file_path,newline="") as f:
            #csv_reader = csv.reader(f,delimiter=delimiter,quotechar=quotechar)
            tf.print("正在读取{}".format(file_path))
            x_list = []
            y_list = []

            #for i,row in enumerate(csv_reader):
            for i,row in enumerate(f):
                row = json.loads(row.strip())
                if i%100000 == 0:
                    tf.print("正在读取第{}行".format(i))
                x_list.append([word2index_dict[str(word)] for word in row[:-1]])
                y_list.append(row[-1])
            res.append(np.array(x_list))
            res.append(np.array(y_list))
            tf.print("读取完成{}".format(file_path))
    return res



def build_model(sentence_maxlen,word_count,embedding_dim,embeddings_matrix=None, is_embeddings_training = False):
    """构建模型
    """
    classify_transformer = ClassifyTransformer(
        num_layers = 4, #4,#encoder_layer的个数
        embedding_dim = embedding_dim,
        num_heads = 8, # 多头的个数，必须是embedding_dim的约数
        dff = embedding_dim * 4, #embedding_dim * 4, # 前馈神经网络的第一层的units
        input_vocab_size = word_count,
        dropout_rate = 0.1, #0.1,
        sentence_maxlen = sentence_maxlen,
        embeddings_matrix = embeddings_matrix,
        is_embeddings_training = is_embeddings_training,
    )

    return classify_transformer

def train():

    sentence_maxlen = 5
    epochs = 100
    batch_size = 128
    learning_rate = 0.001
    model_dir = r"./model_output"
    check_point_dir = r"./model_callback"
    #检查点目录不存在则创建
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)
    tf_board_dir = os.path.join(check_point_dir,"logs")

    train_data_csv_path = r"train.txt"
    eval_data_csv_path = r"eval.txt"
    test_data_csv_path = r"test.txt"

    model_path = r"./transformer.h5"

    word2index_dict_path = "./word2index_dict.bin"
    word2vector_dict_path = "./word2vector_dict.bin"
    word2vector_file_path = r"/word2vector.bin"
    embeddings_matrix,word2vector_dict,word2index_dict = trans_gensim_word2vec2tf_embedding(word2vector_file_path)
    with open(word2vector_dict_path,"wb") as f:
        pickle.dump(word2vector_dict,f)
    word_count,embedding_dim = embeddings_matrix.shape

    #word_count = 100000
    #embedding_dim = 128
    #[x_train_nparray,y_train_nparray,x_eval_nparray,y_eval_nparray,x_test_nparray,y_test_nparray],word2index_dict = \
    #    get_train_test_dataset_without_embeddings(train_data_csv_path,eval_data_csv_path,test_data_csv_path,word_count)

    x_train_nparray,y_train_nparray,x_eval_nparray,y_eval_nparray,x_test_nparray,y_test_nparray = \
        get_train_test_dataset_with_embedding_matrix(train_data_csv_path,eval_data_csv_path,test_data_csv_path,delimiter = " ",word2index_dict=word2index_dict)

    with open(word2index_dict_path,"wb") as f:
        pickle.dump(word2index_dict,f)

    x_train_nparray = tf.keras.preprocessing.sequence.pad_sequences(x_train_nparray, sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
    x_eval_nparray = tf.keras.preprocessing.sequence.pad_sequences(x_eval_nparray, sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
    x_test_nparray = tf.keras.preprocessing.sequence.pad_sequences(x_test_nparray, sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)

    train_tf_dataset = tf.data.Dataset.from_tensor_slices((x_train_nparray,y_train_nparray)).shuffle(buffer_size=2**13).batch(batch_size)
    eval_tf_dataset = tf.data.Dataset.from_tensor_slices((x_eval_nparray,y_eval_nparray)).shuffle(buffer_size=2**13).batch(batch_size)

    tf.print("batch_count = {}".format(y_train_nparray.shape[0] // batch_size))
    tf.print("len(word2index_dict) = {}".format(len(word2index_dict)))
    tf.print("train_list[0]:{}:{}".format(x_train_nparray[0],y_train_nparray[0]))
    tf.print("eval_list[0]:{}:{}".format(x_eval_nparray[0],y_eval_nparray[0]))
    tf.print("test_list[0]:{}:{}".format(x_test_nparray[0],y_test_nparray[0]))

    #初始化模型
    model = build_model(sentence_maxlen,word_count,embedding_dim,embeddings_matrix,is_embeddings_training=False)

    #自定义优化器
    optimizer = tf.keras.optimizers.Adam(CustomLearningRateSchedule(embedding_dim), beta_1=0.9, beta_2=0.98, epsilon=1e-9,clipnorm=1.0,clipvalue=0.5)
    #optimizer = tf.keras.optimizers.Adam(learning_rate,clipnorm=1.0,clipvalue=0.5)

    #定义观测
    train_epoch_loss_avg = tf.keras.metrics.Mean(name="train_loss")
    train_epoch_bin_accuracy =  tf.keras.metrics.BinaryAccuracy(threshold=0.5,name='train_accuracy')
    train_epoch_precision = tf.keras.metrics.Precision(thresholds=0.5,name='train_precision')
    train_epoch_recall = tf.keras.metrics.Recall(thresholds=0.5,name='train_recall')
    train_epoch_auc =  tf.keras.metrics.AUC(num_thresholds=200,name='train_auc')
    eval_epoch_loss_avg = tf.keras.metrics.Mean(name="eval_loss")
    eval_epoch_bin_accuracy =  tf.keras.metrics.BinaryAccuracy(threshold=0.5,name='eval_accuracy')
    eval_epoch_precision = tf.keras.metrics.Precision(thresholds=0.5,name='eval_precision')
    eval_epoch_recall = tf.keras.metrics.Recall(thresholds=0.5,name='eval_recall')
    eval_epoch_auc =  tf.keras.metrics.AUC(num_thresholds=200,name='eval_auc')

    #定义checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model,)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=check_point_dir, checkpoint_name="model.ckpt", max_to_keep=None)

    #定义tensorboard
    train_summary_writer = tf.summary.create_file_writer(os.path.join(tf_board_dir,"train"))     # 实例化记录器
    eval_summary_writer = tf.summary.create_file_writer(os.path.join(tf_board_dir,"eval"))     # 实例化记录器

    #开启Trace，可以记录图结构和profile信息 #但是一定要记住保存到文件里，否则会一直站用内存
    #tf.summary.trace_on(graph=True, profiler=True)

    #训练模型
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        #训练
        for batch,(x,y) in enumerate(train_tf_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x)
                loss = tf.keras.losses.binary_crossentropy(y_true=tf.reshape(y,y_pred.shape), y_pred=y_pred)
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_epoch_loss_avg.update_state(loss)
            train_epoch_bin_accuracy.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
            train_epoch_precision.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
            train_epoch_recall.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
            train_epoch_auc.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
            if batch % 2**4 == 0:
                tf.print("TRAIN --- BatchSize: {} | Epoch: {:03d} | Batch: {:03d} | BatchLoss: {:.3f} | BatchBinaryAccuracy: {:.3f} | BatchPrecision: {:03f} | BatchRecall: {:03f} | BatchAuc: {:.3f} ".format(
                    batch_size,epoch,batch,
                    loss,
                    tf.keras.metrics.BinaryAccuracy(threshold=0.5)(y_true=y,y_pred=y_pred),
                    tf.keras.metrics.Precision(thresholds=0.5)(y_true=y,y_pred=tf.reshape(y_pred,y.shape)),
                    tf.keras.metrics.Recall(thresholds=0.5)(y_true=y,y_pred=tf.reshape(y_pred,y.shape)),
                    tf.keras.metrics.AUC(num_thresholds=200)(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
                ))

        #验证集上评估一次
        if epoch % 1 == 0:
            #tensorboard记录训练集
            with train_summary_writer.as_default():   # 指定记录器
                tf.summary.scalar("AverageLoss", train_epoch_loss_avg.result(), step=epoch)       # 将当前损失函数的值写入记录器
                tf.summary.scalar("AverageBinaryAccuracy", train_epoch_bin_accuracy.result(), step=epoch)
                tf.summary.scalar("AveragePrecision", train_epoch_precision.result(), step=epoch)
                tf.summary.scalar("AverageRecall", train_epoch_recall.result(), step=epoch)
                tf.summary.scalar("AverageAuc", train_epoch_auc.result() , step=epoch)
            #验证集评估
            for batch,(x,y) in enumerate(eval_tf_dataset):
                y_pred = model(x)
                loss = tf.keras.losses.binary_crossentropy(y_true=tf.reshape(y,y_pred.shape), y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                eval_epoch_loss_avg.update_state(loss)
                eval_epoch_bin_accuracy.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
                eval_epoch_precision.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
                eval_epoch_recall.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))
                eval_epoch_auc.update_state(y_true=y,y_pred=tf.reshape(y_pred,y.shape))

            tf.print("BatchSize: {} | Epoch: {:03d} \nTRAIN --- BAverageLoss: {:.3f} | AverageBinaryAccuracy: {:.3f} | AveragePrecision: {:03f} | AverageRecall: {:03f} | AverageAuc: {:.3f} \nEVALUATE---AverageLoss: {:.3f} | AverageBinaryAccuracy: {:.3f} | AveragePrecision: {:03f} | AverageRecall: {:03f} | AverageAuc: {:.3f} ".format(
                batch_size,epoch,
                train_epoch_loss_avg.result(),
                train_epoch_bin_accuracy.result(),
                train_epoch_precision.result(),
                train_epoch_recall.result(),
                train_epoch_auc.result(),

                eval_epoch_loss_avg.result(),
                eval_epoch_bin_accuracy.result(),
                eval_epoch_precision.result(),
                eval_epoch_recall.result(),
                eval_epoch_auc.result()
            ))
            path = checkpoint_manager.save(checkpoint_number=epoch)
            with eval_summary_writer.as_default():   # 指定记录器
                tf.summary.scalar("AverageLoss", eval_epoch_loss_avg.result(), step=epoch)       # 将当前损失函数的值写入记录器
                tf.summary.scalar("AverageBinaryAccuracy", eval_epoch_bin_accuracy.result(), step=epoch)
                tf.summary.scalar("AveragePrecision", eval_epoch_precision.result(), step=epoch)
                tf.summary.scalar("AverageRecall", eval_epoch_recall.result(), step=epoch)
                tf.summary.scalar("AverageAuc", eval_epoch_auc.result() , step=epoch)

            tf.print("Save checkpoint to path: {}".format(path))
            tf.print("This epoch spends {:.1f}s".format(time.time()-start_time))
            train_epoch_loss_avg.reset_states()
            train_epoch_bin_accuracy.reset_states()
            train_epoch_precision.reset_states()
            train_epoch_recall.reset_states()
            train_epoch_auc.reset_states()

            eval_epoch_loss_avg.reset_states()
            eval_epoch_bin_accuracy.reset_states()
            eval_epoch_precision.reset_states()
            eval_epoch_recall.reset_states()
            eval_epoch_auc.reset_states()

    #保存模型
    tf.saved_model.save(model,model_dir)

    test_predict_probility_list = model(x_test_nparray)
    eval_predict_probility_list = model(x_eval_nparray)

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
        tf.print("{} sklearn.metrics.classification_report:".format(interval))
        tf.print(sklearn.metrics.classification_report(y_true=real_probility_nparray,y_pred=predict_probility_trans_nparray,digits=4))


def predict_model(input_sentence_file_path,output_sentence_file_path):

    sentence_maxlen = 5
    batch_size = 2**17

    model_path = r"./weights.06-0.17.h5"
    model =  tf.keras.models.load_model(model_path)

    word2index_dict_path = "./word2index_dict.bin"
    with open(word2index_dict_path,"rb") as f:
        word2index_dict = pickle.load(f)
    tf.print({"<UNK>":word2index_dict["<UNK>"],"<PADDING>":word2index_dict["<PADDING>"]})

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
                tf.print("正在预测{}行".format(i))
                tf.print("检查输入")
                tf.print(input_sentence_list[0])
                trans_input_sentence_nparray = np.array([[word2index_dict[str(v)] for v in row] for row in input_sentence_list])
                tf.print(trans_input_sentence_nparray[0])
                trans_input_sentence_nparray = tf.keras.preprocessing.sequence.pad_sequences(trans_input_sentence_nparray,
                    sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
                tf.print("x_input的长度为{}".format(len(trans_input_sentence_nparray)))
                predict_probility_list = model.predict(
                    trans_input_sentence_nparray,
                    verbose=0,
                    batch_size=int(batch_size),
                    workers=multiprocessing.cpu_count()*5,
                    use_multiprocessing=True,
                )
                for i,(x,y) in enumerate(zip(input_sentence_list,predict_probility_list)):
                    if i % int(batch_size) == 0:
                        tf.print("正在写入该batch第{}行".format(i))
                    x.extend(y)
                    csv_writer.writerow(x)
                #每个batch都要被清空
                input_sentence_list = []
        if input_sentence_list:
            tf.print("检查输入")
            tf.print(input_sentence_list[0])
            trans_input_sentence_nparray = np.array([[word2index_dict[str(v)] for v in row] for row in input_sentence_list])
            tf.print(trans_input_sentence_nparray[0])
            trans_input_sentence_nparray = tf.keras.preprocessing.sequence.pad_sequences(trans_input_sentence_nparray,
                sentence_maxlen,dtype=int,truncating="post", padding='post',value=0)
            tf.print(len(trans_input_sentence_nparray))
            predict_probility_list = model.predict(
                trans_input_sentence_nparray,
                verbose=0,
                batch_size=int(batch_size),
                workers=multiprocessing.cpu_count()*5,
                use_multiprocessing=True,
            )
            for i,(x,y) in enumerate(zip(input_sentence_list,predict_probility_list)):
                if i % int(batch_size) == 0:
                    tf.print("x_input的长度为{}".format(len(trans_input_sentence_nparray)))
                x.extend(y)
                csv_writer.writerow(x)
            input_sentence_list = []

if __name__ == "__main__":
    train()
    #predict_model(os.path.join(APP_DIR,"predicts/predict_three_node.csv"),"./three_node.csv")
    #predict_one()
