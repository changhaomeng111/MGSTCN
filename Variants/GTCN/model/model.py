import tensorflow as tf
from model.mgstcn_cell import MGSTCNCell
from model.dilate_cell import DilatedCell
from tensorflow.contrib.rnn import LSTMStateTuple
class MGSTCNModel(object):
    def __init__(self, is_training,batch_size, scaler, adj_mx, **model_kwargs):

        self._scaler = scaler
        self._loss = None
        self._mae = None
        self._train_op = None

        k = int(model_kwargs.get('k', 2))
        horizon = int(model_kwargs.get('horizon', 1))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 5000))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))

        ratelist=[1,2,3]
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        cell = DilatedCell(rnn_units,num_nodes)
        EncodeCell = MGSTCNCell(rnn_units, adj_mx, k=k, num_nodes=num_nodes,ratelist=ratelist)  #,num_proj=horizon
        DecodeCell = MGSTCNCell(rnn_units, adj_mx, k=k, num_nodes=num_nodes, ratelist=ratelist, num_proj=output_dim)

        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('INPUTS_SKIP'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            Pinputs=inputs
            labels_offset=inputs[-1]

            for rate in ratelist:
                n_steps = len(Pinputs)
                dialated_n_steps = n_steps // rate
                dilated_inputs = [tf.concat(Pinputs[i * rate:(i + 1) * rate], axis=0) for i in range(dialated_n_steps)]  #
                step_length=len(dilated_inputs)

                globals()["State_list" + str(rate)] = []
                scope_name = "Dilated_%d" % rate
                with tf.variable_scope(scope_name):
                    for i in range (0,step_length):
                        if i==0:
                            zero_tensor=tf.zeros([batch_size * rate, rnn_units * num_nodes], tf.float32)
                            pre_state = LSTMStateTuple(zero_tensor,zero_tensor)
                        _, dilated_inputs_state= cell(dilated_inputs[i], pre_state)
                        pre_state=dilated_inputs_state
                        globals()["State_list" + str(rate)].append(dilated_inputs_state)

            for i in range(0, len(inputs)):
                if i == 0:
                    state_c = tf.zeros([batch_size, rnn_units * num_nodes])
                    state_h = tf.zeros([batch_size, rnn_units * num_nodes])
                    for rate in ratelist:
                        state_init = tf.zeros([rate * batch_size, rnn_units * num_nodes])
                        state_c = tf.concat([state_c, state_init], axis=0)
                        state_h = tf.concat([state_h, state_init], axis=0)
                    main_state = LSTMStateTuple(state_c, state_h)
                else:
                    state_c = main_state[0]
                    state_h = main_state[1]
                    for rate in ratelist:
                        if int(i / rate):
                            read_state = globals()["State_list" + str(rate)][int(i / rate) - 1]
                            state_c = tf.concat([state_c, read_state[0]], axis=0)
                            state_h = tf.concat([state_h, read_state[1]], axis=0)
                        else:
                            state_init = tf.zeros([rate * batch_size, rnn_units * num_nodes])
                            state_c = tf.concat([state_c, state_init], axis=0)
                            state_h = tf.concat([state_h, state_init], axis=0)
                    main_state = LSTMStateTuple(state_c, state_h)

                _, cell_state = EncodeCell(inputs[i], main_state)
                main_state=cell_state

            self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim),name='labels')
            labels = tf.unstack(tf.reshape(self._labels, (batch_size, horizon, num_nodes * output_dim)), axis=1)
            #labels_offset = tf.zeros(shape=(batch_size, num_nodes * output_dim))
            labels.insert(0,labels_offset)
            labels=labels[:-1]
            Pinputs = labels

            for rate in ratelist:
                n_steps = len(Pinputs)
                dialated_n_steps = n_steps // rate
                dilated_inputs = [tf.concat(Pinputs[i * rate:(i + 1) * rate], axis=0) for i in range(dialated_n_steps)]
                step_length = len(dilated_inputs)

                globals()["Test_State_list" + str(rate)] = []
                with tf.variable_scope("Test_Dilation_%d" % rate):
                    for i in range(0, step_length):
                        if i == 0:
                            pre_test_state = globals()["State_list" + str(rate)][-1]
                            pre_test_state = LSTMStateTuple(pre_test_state[0], pre_test_state[1])
                        _, dilated_cell_state = cell(dilated_inputs[i], pre_test_state)
                        pre_test_state=dilated_cell_state
                        globals()["Test_State_list" + str(rate)].append(dilated_cell_state)

            def loop_function(prev, i):
                if is_training:
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    result = prev
                return result

            output_list = []
            prev = None
            for i, inp in enumerate(labels):
                if i == 0:
                    state_c = cell_state[0]
                    state_h = cell_state[1]
                    for rate in ratelist:
                        read_state=globals()["State_list" + str(rate)][-1]
                        state_c = tf.concat([state_c, read_state[0]], axis=0)
                        state_h = tf.concat([state_h, read_state[1]], axis=0)
                    main_state = LSTMStateTuple(state_c, state_h)
                else:
                    state_c=main_state[0]
                    state_h=main_state[1]
                    for rate in ratelist:
                        if int(i / rate):
                            read_state=globals()["Test_State_list" + str(rate)][int(i / rate) - 1]
                            state_c = tf.concat([state_c, read_state[0]], axis=0)
                            state_h = tf.concat([state_h, read_state[1]], axis=0)
                        else:
                            state_c1 = tf.zeros([rate * batch_size, rnn_units * num_nodes])
                            state_h1 = tf.zeros([rate * batch_size, rnn_units * num_nodes])
                            state_c = tf.concat([state_c, state_c1], axis=0)
                            state_h = tf.concat([state_h, state_h1], axis=0)
                    main_state = LSTMStateTuple(state_c, state_h)

                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = DecodeCell(inp, main_state)
                main_state=state
                output_list.append(output)
                prev = output

            outputs = tf.stack(output_list, axis=1)
            self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)
    @property
    def inputs(self):
        return self._inputs
    @property
    def labels(self):
        return self._labels
    @property
    def loss(self):
        return self._loss
    @property
    def mae(self):
        return self._mae
    @property
    def outputs(self):
        return self._outputs
