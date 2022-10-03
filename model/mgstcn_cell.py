import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.contrib.rnn import RNNCell

class MGSTCNCell(RNNCell):

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, k, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, use_gc_for_ru=True):

        super(MGSTCNCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._num_proj = num_proj
        self._k = k
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        adj_mx = sp.csr_matrix(adj_mx)
        M, _ = adj_mx.shape
        I = sp.identity(M, format='csr', dtype=adj_mx.dtype)
        adj_mx = (2 / 2 * adj_mx) - I
        adj_mx = adj_mx.astype(np.float32)
        supports.append(adj_mx)
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope("Cell"):
            with tf.variable_scope("gates"):
                output_size = self._num_units * 4
                value = self._gconv(inputs,state, output_size, bias_start=1.0)
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                i, j, f, o = tf.split(value=value, num_or_size_splits=4, axis=-1)
                i = tf.reshape(i, (-1, self._num_nodes * self._num_units))
                j = tf.reshape(j, (-1, self._num_nodes * self._num_units))
                f = tf.reshape(f, (-1, self._num_nodes * self._num_units))
                o = tf.reshape(o, (-1, self._num_nodes * self._num_units))
            with tf.variable_scope("candidate"):
                c = self._gconv(inputs,state, self._num_units)
                new_c = self._activation(c) * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * self._activation(j)
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = new_h
            output=new_h
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_h, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return tf.reshape(value, [batch_size, self._num_nodes * output_size])

    def _gconv(self, inputs, state, output_size, bias_start=0.0):

        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype
        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        with tf.variable_scope("share"):
            if self._k == 0:
                pass
            else:
                for support in self._supports:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._k+ 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(self._supports) * self._k + 1
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable('weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)

            biases = tf.get_variable('biases', [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)

        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
