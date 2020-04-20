import tensorflow as tf
import tensorflow_addons as tf_ad


class BiLstmCrfModel(tf.keras.Model):
    """bi_lstm_crf主要模型

    """
    def __init__(self,
        lstm_unit_num, vocab_size, tag_size,
        embedding_dim,embedding_matrix=None, is_embedding_training=True,
        embedding_dropout_rate = 0.0,
    ):
        super().__init__()
        self.lstm_unit_num = lstm_unit_num
        self.vocab_size = vocab_size
        self.tag_size = tag_size

        if not (embedding_matrix is None):
            self.embedding = tf.keras.layers.Embedding(
                input_dim = vocab_size,
                output_dim = embedding_dim,
                embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                trainable = is_embedding_training,
            )
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim = vocab_size,
                output_dim = embedding_dim,
                trainable = is_embedding_training,
            )


        self.embedding_dropout = tf.keras.layers.Dropout(embedding_dropout_rate)

        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = lstm_unit_num,
                activation='tanh',
                dropout=0.0,
                recurrent_dropout=0.0,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                unit_forget_bias=True,
                return_sequences=True
            )
        )
        self.dense = tf.keras.layers.Dense(units = tag_size,activation=None,use_bias=True)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(tag_size, tag_size)),
                                             trainable=False)


    def call(self, x_batch,y_batch=None,training=None):
        #<PADDING> = 0
        x_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(x_batch, 0), dtype=tf.int32), axis=-1)

        #None,maxlen
        inputs = self.embedding(x_batch)
        #None,maxlen,embedding_dim
        inputs = self.embedding_dropout(inputs = inputs, training = training)
        inputs = self.bi_lstm(inputs)
        #None,maxlen,lstm_unit_num * 2
        logits = self.dense(inputs)
        #None,maxlen,tag_size

        if not (y_batch is None):
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, y_batch, x_length)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, x_length, log_likelihood
        else:
            return logits, x_length


