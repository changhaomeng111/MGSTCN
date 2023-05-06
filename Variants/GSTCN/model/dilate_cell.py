from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple


class DilatedCell(RNNCell):


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

    def __init__(self, num_units,num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=True,state_is_tuple=True):
        """

        :param num_units:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".

        """
        super(DilatedCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._num_proj=num_proj
        self._state_is_tuple=state_is_tuple

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_nodes * self._num_units, self._num_nodes * self._num_units)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):

        c, h = state
        with tf.variable_scope(scope or "DilatedCell",reuse=tf.AUTO_REUSE ):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                    value =  self._fc(inputs=inputs,state=h,output_size=4* self._num_units, bias_start=0.0)
                    _i,_j,_f,_o = tf.split(value=value, num_or_size_splits=4, axis=-1)  #5 64 20800
                    _i = self.ln(_i, scope='i/')
                    _j = self.ln(_j, scope='j/')
                    _f= self.ln(_f, scope='f/')
                    _o = self.ln(_o, scope='o/')

            with tf.variable_scope("candidate"):
                c = self._fc(inputs=inputs,state=c,output_size=self._num_units, bias_start=0.0)
                new_c =self._activation(c) * tf.nn.sigmoid(_f) + tf.nn.sigmoid(_i) * self._activation(_j)

            new_h = self._activation(self.ln(new_c)) * tf.nn.sigmoid(_o)
            new_state = LSTMStateTuple(new_c,new_h)

            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    biases = tf.get_variable('b', [self._num_proj], dtype='float32',
                                             initializer=tf.constant_initializer(0.0, dtype='float32'))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_h, shape=(-1, self._num_units))
                    new_h = tf.reshape(tf.nn.bias_add(tf.matmul(output, w), biases),
                                       shape=(batch_size, self.output_size))

            return new_h, new_state

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size,self._num_nodes, -1))
        state = tf.reshape(state, (batch_size ,self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.matmul(inputs_and_state, weights)
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return tf.reshape(value, [batch_size, self._num_nodes * output_size])
