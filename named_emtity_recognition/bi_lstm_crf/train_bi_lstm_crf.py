import tensorflow as tf
import tensorflow_addons as tf_ad
import os
import time

APP_DIR  = os.path.dirname(os.path.realpath(__file__))
from utils import trans_gensim_word2vec2tf_embedding,trans_data2tf_data
from model import BiLstmCrfModel

def split_train_eval_test_dataset(dataset,batch_size):
    """区分训练验证测试集
    """
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    tf.print("总共有数据{}条".format(dataset_size))
    dataset = dataset.shuffle(buffer_size=2**13)
    train_size = int(0.7 * dataset_size)
    eval_size = int(0.15 * dataset_size)
    test_size = int(0.15 * dataset_size)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    eval_dataset = test_dataset.skip(eval_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), \
        eval_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), \
        test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

#@tf.function
def train_one_step(x_batch,y_batch,is_return_accuracy=True,):
    """一步训练
    """
    with tf.GradientTape() as tape:
        logit_batch , x_length_batch , log_likelihood = model(x_batch, y_batch,training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_epoch_loss_avg(loss)
    accuracy = 0
    if is_return_accuracy:
        accuracy = calculate_accuracy_one_step(logit_batch , x_length_batch , log_likelihood)
    train_epoch_accuracy_avg(accuracy)
    return loss,accuracy

def eval_one_step(x_batch,y_batch,is_return_accuracy=True):
    """一步验证
    """
    logit_batch , x_length_batch , log_likelihood = model(x_batch, y_batch,training=True)
    loss = - tf.reduce_mean(log_likelihood)
    eval_epoch_loss_avg(loss)
    accuracy = 0
    if is_return_accuracy:
        accuracy = calculate_accuracy_one_step(logit_batch , x_length_batch , log_likelihood)
    eval_epoch_accuracy_avg(accuracy)
    return loss,accuracy



def calculate_accuracy_one_step(logit_batch , x_length_batch , log_likelihood):
    """计算一步的准确度
    """
    accuracy = 0
    path_list = []
    for logit, x_length, y in zip(logit_batch, x_length_batch, y_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:x_length], model.transition_params)
        assert len(viterbi_path) == x_length
        path_list.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(viterbi_path,dtype=tf.int32),
            tf.convert_to_tensor(y[:x_length],dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = accuracy / len(path_list)
    return accuracy


if __name__ == "__main__":
    """
    训练模型
    """
    #word2vec路径
    word2vec_file_path = os.path.join(APP_DIR,"word2vec/word2vector.bin")
    #data文件路径
    data_path = os.path.join(APP_DIR,"data/ner_input.txt")
    #训练次数
    epochs = 2
    #批大小
    batch_size = 128
    #学习率
    learning_rate = 1e-3
    #embedding layer是否参加训练
    is_embedding_training = False

    ###以上是需要修改的部分


    model_dir = os.path.join(APP_DIR,"model_output/")
    check_point_dir = os.path.join(APP_DIR,"model_callback/")
    #检查点目录不存在则创建，防止tensorflowboard报错
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)
    tf_board_dir = os.path.join(check_point_dir,"logs")


    embedding_matrix,word2vector_dict,word2index_dict = trans_gensim_word2vec2tf_embedding(word2vec_file_path)
    vocab_size,embedding_dim = embedding_matrix.shape

    #划分训练集，验证集，测试集
    dataset,tag2index_dict = trans_data2tf_data(data_path,word2index_dict)
    train_dataset,eval_dataset,test_dataset = split_train_eval_test_dataset(dataset,batch_size)
    tag_size = len(tag2index_dict)

    #建立模型
    model = BiLstmCrfModel(lstm_unit_num=embedding_dim,vocab_size=vocab_size,tag_size=tag_size,
        embedding_dim=embedding_dim,embedding_matrix=embedding_matrix,is_embedding_training=is_embedding_training)

    #自定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate,clipnorm=5.0,clipvalue=0.5)

    #定义观测
    train_epoch_loss_avg = tf.keras.metrics.Mean(name="train_loss")
    train_epoch_accuracy_avg = tf.keras.metrics.Mean(name="train_accuracy")
    eval_epoch_loss_avg = tf.keras.metrics.Mean(name="eval_loss")
    eval_epoch_accuracy_avg = tf.keras.metrics.Mean(name="eval_accuracy")

    #定义checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model,)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=check_point_dir, checkpoint_name="model.ckpt", max_to_keep=None)

    #定义tensorboard
    train_summary_writer = tf.summary.create_file_writer(os.path.join(tf_board_dir,"train"))     # 实例化记录器
    eval_summary_writer = tf.summary.create_file_writer(os.path.join(tf_board_dir,"eval"))     # 实例化记录器

    #训练模型
    for epoch in range(1, epochs + 1):
        for batch,(x_batch,y_batch) in enumerate(train_dataset):
            loss,accuracy = train_one_step(x_batch,y_batch)
            if batch % 2**4 == 0:
                tf.print("TRAIN --- BatchSize: {} | Epoch: {:03d} | Batch: {:03d} | BatchLoss: {:.3f} | BatchAccuracy: {:.3f}".format(batch_size,epoch,batch,loss,accuracy))

        #验证集上评估一次
        if epoch % 1 == 0:
            start_time = time.time()
            #tensorboard记录训练集
            with train_summary_writer.as_default():   # 指定记录器
                tf.summary.scalar("AverageLoss", train_epoch_loss_avg.result(), step=epoch)       # 将当前损失函数的值写入记录器
                tf.summary.scalar("AverageAccuracy", train_epoch_accuracy_avg.result(), step=epoch)       # 将当前损失函数的值写入记录器
            #验证集评估
            for batch,(x_batch,y_batch) in enumerate(eval_dataset):
                loss,accuracy = eval_one_step(x_batch,y_batch)
            tf.print("BatchSize: {} | Epoch: {:03d} \nTRAIN --- AverageLoss: {:.3f} | AverageAccuracy: {:.3f} \nEVALUATE---AverageLoss: {:.3f} | AverageAccuracy: {:.3f}".format(
                batch_size,epoch,
                train_epoch_loss_avg.result(),
                train_epoch_accuracy_avg.result(),
                eval_epoch_loss_avg.result(),
                eval_epoch_accuracy_avg.result(),
            ))
            #tensorboard记录验证集
            with eval_summary_writer.as_default():   # 指定记录器
                tf.summary.scalar("AverageLoss", eval_epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar("AverageAccuracy", eval_epoch_accuracy_avg.result(), step=epoch)       # 将当前损失函数的值写入记录器
            path = checkpoint_manager.save(checkpoint_number=epoch)
            tf.print("Save checkpoint to path: {}".format(path))
            tf.print("This epoch spends {:.1f}s".format(time.time()-start_time))
            train_epoch_loss_avg.reset_states()
            train_epoch_accuracy_avg.reset_states()
            eval_epoch_loss_avg.reset_states()
            eval_epoch_accuracy_avg.reset_states()
