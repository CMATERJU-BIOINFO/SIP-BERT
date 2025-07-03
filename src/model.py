import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class GlobalAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_key, d_value, **kwargs):
        self.n_heads = n_heads
        self.d_key = d_key
        self.sqrt_d_key = tf.sqrt(tf.cast(d_key, tf.float32))
        self.d_value = d_value
        self.d_output = n_heads * d_value
        super(GlobalAttention, self).__init__(**kwargs)

    def build(self, input_shapes):
        (_, self.d_global_input), (_, _, self.d_seq_input) = input_shapes
        self.Wq = self.add_weight(name='Wq', shape=(self.n_heads, self.d_global_input, self.d_key),
                                  initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(name='Wk', shape=(self.n_heads, self.d_seq_input, self.d_key),
                                  initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(name='Wv', shape=(self.n_heads, self.d_seq_input, self.d_value),
                                  initializer='glorot_uniform', trainable=True)
        super(GlobalAttention, self).build(input_shapes)

    def call(self, inputs):
        X, S = inputs
        _, length, _ = K.int_shape(S)

        # Value transformation
        VS = tf.transpose(tf.keras.activations.gelu(
            tf.tensordot(S, self.Wv, axes=[[2], [1]])
        ), perm=[0, 2, 1, 3])
        VS_batched_heads = tf.reshape(VS, (-1, length, self.d_value))

        # Attention weights
        QX = tf.tanh(tf.tensordot(X, self.Wq, axes=[[1], [1]]))
        QX_batched_heads = tf.reshape(QX, (-1, self.d_key))
        
        KS = tf.transpose(tf.tanh(
            tf.tensordot(S, self.Wk, axes=[[2], [1]])
        ), perm=[0, 2, 3, 1])
        KS_batched_heads = tf.reshape(KS, (-1, self.d_key, length))
        
        attn_logits = tf.matmul(
            tf.expand_dims(QX_batched_heads, 1), 
            KS_batched_heads
        ) / self.sqrt_d_key
        Z_batched_heads = tf.nn.softmax(tf.squeeze(attn_logits, axis=1))

        # Attention output
        Y_batched_heads = tf.matmul(
            tf.expand_dims(Z_batched_heads, 1), 
            VS_batched_heads
        )
        Y = tf.reshape(Y_batched_heads, (-1, self.d_output))
        return Y

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], self.d_output)

def create_model(seq_len, vocab_size, n_annotations, d_hidden_seq=128, d_hidden_global=512, 
                n_blocks=6, n_heads=4, d_key=64, conv_kernel_size=9, wide_conv_dilation_rate=5, 
                activation='gelu', dropout_rate=0.5):
    
    d_value = d_hidden_global // n_heads

    input_seq = keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name='input-seq')
    input_annotations = keras.layers.Input(shape=(n_annotations,), dtype=tf.float32, name='input-annotations')

    hidden_seq = keras.layers.Embedding(vocab_size, d_hidden_seq, name='embedding-seq-input')(input_seq)
    hidden_global = keras.layers.Dense(d_hidden_global, activation=activation, name='dense-global-input')(input_annotations)

    for block_index in range(1, n_blocks + 1):
        # Sequence processing block
        seqed_global = keras.layers.Dense(d_hidden_seq, activation=activation, 
                                         name=f'global-to-seq-dense-block{block_index}')(hidden_global)
        seqed_global = keras.layers.Reshape((1, d_hidden_seq), 
                                          name=f'global-to-seq-reshape-block{block_index}')(seqed_global)

        narrow_conv_seq = keras.layers.Conv1D(filters=d_hidden_seq, kernel_size=conv_kernel_size,
                                             padding='same', dilation_rate=1, activation=activation,
                                             name=f'narrow-conv-block{block_index}')(hidden_seq)
        narrow_conv_seq = keras.layers.Dropout(dropout_rate)(narrow_conv_seq)

        wide_conv_seq = keras.layers.Conv1D(filters=d_hidden_seq, kernel_size=conv_kernel_size,
                                           padding='same', dilation_rate=wide_conv_dilation_rate,
                                           activation=activation, name=f'wide-conv-block{block_index}')(hidden_seq)
        wide_conv_seq = keras.layers.Dropout(dropout_rate)(wide_conv_seq)

        hidden_seq = keras.layers.Add(name=f'seq-merge1-block{block_index}')([hidden_seq, seqed_global, narrow_conv_seq, wide_conv_seq])
        hidden_seq = keras.layers.LayerNormalization(name=f'seq-merge1-norm-block{block_index}')(hidden_seq)

        dense_seq = keras.layers.Dense(d_hidden_seq, activation=activation, 
                                     name=f'seq-dense-block{block_index}')(hidden_seq)
        dense_seq = keras.layers.Dropout(dropout_rate)(dense_seq)
        hidden_seq = keras.layers.Add(name=f'seq-merge2-block{block_index}')([hidden_seq, dense_seq])
        hidden_seq = keras.layers.LayerNormalization(name=f'seq-merge2-norm-block{block_index}')(hidden_seq)

        # Global processing block
        dense_global = keras.layers.Dense(d_hidden_global, activation=activation, 
                                        name=f'global-dense1-block{block_index}')(hidden_global)
        dense_global = keras.layers.Dropout(dropout_rate)(dense_global)
        attention = GlobalAttention(n_heads, d_key, d_value, 
                                  name=f'global-attention-block{block_index}')([hidden_global, hidden_seq])
        hidden_global = keras.layers.Add(name=f'global-merge1-block{block_index}')([hidden_global, dense_global, attention])
        hidden_global = keras.layers.LayerNormalization(name=f'global-merge1-norm-block{block_index}')(hidden_global)

        dense_global = keras.layers.Dense(d_hidden_global, activation=activation, 
                                        name=f'global-dense2-block{block_index}')(hidden_global)
        dense_global = keras.layers.Dropout(dropout_rate)(dense_global)
        hidden_global = keras.layers.Add(name=f'global-merge2-block{block_index}')([hidden_global, dense_global])
        hidden_global = keras.layers.LayerNormalization(name=f'global-merge2-norm-block{block_index}')(hidden_global)

    output_layer = keras.layers.Dense(1, activation='sigmoid', name='output-classification')(hidden_global)
    return keras.models.Model(inputs=[input_seq, input_annotations], outputs=output_layer)