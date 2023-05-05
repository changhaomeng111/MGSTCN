from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.sparse as sp
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple
from lib.utils import *
import numpy as np
from lib import utils

class MGSTCNCell(RNNCell):
    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass
    def ln(self,tensor, scope=None, epsilon=1e-5):
        """ Layer normalizes a 2D tensor"""
        assert (len(tensor.get_shape()) == 2)
        m, v = tf.nn.moments(tensor, [1], keep_dims=True)
        if not isinstance(scope, str):
            scope = ''
        with tf.variable_scope(scope + 'layer_norm'):
            scale = tf.get_variable('scale',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(1))
            shift = tf.get_variable('shift',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(0))
        ln_initial = (tensor - m) / tf.sqrt(v + epsilon)
        return ln_initial * scale + shift

    def calculate_scaled_laplacian(self,adj_mx, lambda_max=2):
        L = sp.csr_matrix(adj_mx)
        M, _ = L.shape
        I = sp.identity(M, format='csr', dtype=L.dtype)
        L = (2 / lambda_max * L) - I
        return L.astype(np.float32)

    def __init__(self, num_units, adj_mx, k, num_nodes, ratelist=[],num_proj=None,
                 activation=tf.nn.tanh, reuse=True, state_is_tuple=True):

        super(MGSTCNCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._rate =ratelist
        self._num_units = num_units
        self._num_proj=num_proj
        self._max_diffusion_step = k
        self._supports = []
        self._state_is_tuple=state_is_tuple
        supports = []
        supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        d_adp=(len(ratelist) + 1)*(len(supports) )

        with tf.variable_scope("adp", reuse=tf.AUTO_REUSE):
            self.nodevec1 = tf.get_variable('nodevec1', [d_adp, d_adp], dtype='float32',initializer=tf.contrib.layers.xavier_initializer(1))
            self.nodevec2 = tf.get_variable('nodevec2', [d_adp, d_adp], dtype='float32',initializer=tf.contrib.layers.xavier_initializer(1))
            self.adp = tf.nn.softmax(tf.nn.relu(tf.matmul(self.nodevec1, self.nodevec2)), dim=1)

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
        return LSTMStateTuple(self._num_nodes * self._num_units, self._num_nodes * self._num_units)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size



    def __call__(self, inputs, expansion_state, scope=None):
        self.batch_size = inputs.get_shape()[0].value
        expansion_c, expansion_h = expansion_state
        c = expansion_c[:self.batch_size, :]  # shape=(64, 325*64)
        h = expansion_h[:self.batch_size,:]    #shape=(64, 325*64)
        state_len=expansion_h.get_shape()[0].value

        if state_len>self.batch_size:
            offset = self.batch_size

            for i in range  (0,len(self._rate)):
                globals()["ct" + str(i)] = expansion_c[offset:(offset + self._rate[i]*self.batch_size), :][:self.batch_size, :]
                globals()["ht" + str(i)]=expansion_h[offset:(offset+self._rate[i]*self.batch_size),:][:self.batch_size, :]
                offset+=self._rate[i]*self.batch_size


        with tf.variable_scope(scope or "mgstcn_cell2",reuse=tf.AUTO_REUSE):
            with tf.variable_scope("gates"):

                for i in range(0, len(self._supports)):
                    value=(self._fc(inputs,h,output_size= self._num_units*(4+ len(self._rate))))

                for i in range(0,len(self._supports)):
                    for j in range(0, len(self._rate)):
                        with tf.variable_scope(str(i)+"modular" + str(j)):
                            value+=self._fc(inputs,globals()["ht" + str(j)], output_size=self._num_units*(4+ len(self._rate)))

                value = tf.split(value=value, num_or_size_splits=(4+ len(self._rate)), axis=-1)
                _i = self.ln(value[0],scope='i')
                _j = self.ln(value[1],scope='j')
                _f = self.ln(value[2],scope='f')
                _o = self.ln(value[3],scope='o')
                for i in range(0, len(self._rate)):
                    globals()["new_f" + str(i)] =self.ln( value[i + 4],scope='new_f' + str(i))

            with tf.variable_scope("candidate"):
                c=self._activation(self._fc(inputs, c,self._num_units,bias_start=0.0)) * tf.nn.sigmoid(_f)
                for i in range(0, len(self._rate)):
                    with tf.variable_scope("c_fusion" + str(i)):
                        c += self._activation(self._fc(inputs, globals()["ct" + str(i)],self._num_units,bias_start=0.0)) * tf.nn.sigmoid(globals()["new_f" + str(i)])
                new_c = c + tf.nn.sigmoid(_i) * self._activation(_j)

            new_h = self._activation(new_c) * tf.nn.sigmoid(_o)
            new_state = LSTMStateTuple(new_c,new_h)
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    biases = tf.get_variable('b', [self._num_proj], dtype='float32',
                                             initializer=tf.constant_initializer(0.0, dtype='float32'))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_h, shape=(-1, self._num_units))
                    new_h = tf.reshape(tf.nn.bias_add(tf.matmul(output, w), biases),shape=(batch_size, self.output_size))

            return new_h, new_state



    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs,state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size ,self._num_nodes, -1))
        state = tf.reshape(state, (batch_size,self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)

        input_size = inputs_and_state.get_shape()[2].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(1))
        value = tf.matmul(inputs_and_state, weights)
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return tf.reshape(value, [batch_size, self._num_nodes * output_size])

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in self._supports:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])

    def _ModularGraphConv(self, inputs=None,h=None, output_size=None, gname="Modular_gconv"):

        batch_size = inputs.get_shape()[0].value  #64
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))   #64*325*2

        h = tf.reshape(globals()["v_10"], (batch_size, self._num_nodes, -1))   #64*325*64
        inputs_and_state = tf.concat([inputs, h], axis=2)   #64*325*66
        x = inputs_and_state

        inputs_and_state_list=[]
        for i in range(0, len(self._supports)):
            h1=globals()["v_1" + str(i)]
            h1 = tf.reshape(h1, (batch_size, self._num_nodes, -1))
            inputs_and_state_list.append(tf.concat([inputs, h1], axis=2))  # 3*64*325*66

        for i in range(0, len(self._supports)):
            for j in range(0, len(self._rate)):

                h2=globals()[str(i) + "v_2" + str(j)]
                h2 = tf.reshape(h2, (batch_size, self._num_nodes, -1))
                inputs_and_state_list.append(tf.concat([inputs, h2], axis=2))  #3*64*325*66

        input_size = (inputs_and_state_list[0]).get_shape()[2].value #66
        dtype = inputs.dtype

        x0 = tf.reshape(inputs_and_state_list, shape=[-1,self._num_nodes*input_size * batch_size])  #325*[66*64]
        x = tf.expand_dims(x, axis=0)
        x = tf.reshape(x, shape=[-1, self._num_nodes * input_size * batch_size])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            x1=tf.matmul(self.adp,x0)
            x =tf.concat([x, x1],axis=0)

            num_matrices = self.adp.shape[0] + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                gname + 'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.constant_initializer(1))
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(gname + 'biases', [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(0))
            x = tf.nn.bias_add(x, biases)

        return tf.reshape(x, [batch_size, self._num_nodes * output_size])


