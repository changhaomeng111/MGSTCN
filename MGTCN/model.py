import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq
from model.mgstcn_cell import MGSTCNCell
from model.mgstcn_decoder import DecoderCell

class MGSTCNModel(object):
    def __init__(self, batch_size, scaler, adj_mx, **model_kwargs):

        self._scaler = scaler
        self._loss = None
        self._mae = None
        self._train_op = None

        k = int(model_kwargs.get('k', 2))
        horizon = int(model_kwargs.get('horizon', 1))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        cell = MGSTCNCell(rnn_units, adj_mx, k=k, num_nodes=num_nodes)
        encoding_cells = [cell] * num_rnn_layers
        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=False)

        cell_with_projection = DecoderCell(rnn_units,num_nodes=num_nodes,num_proj=output_dim)
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=False)
        tf.train.get_or_create_global_step()
        with tf.variable_scope('MGSTGCN_SKIP'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim),name='labels')
            Pinputs=inputs
            Totaloutputlist=[]
            for rate in [1,2,3,4,6]:
                labels2 =  tf.unstack(tf.reshape(self._labels, (batch_size, horizon, num_nodes * output_dim)), axis=1)
                n_steps = len(Pinputs)
                if rate < 0 or rate >= n_steps:
                    raise ValueError('The \'rate\' variable static_rnnneeds to be adjusted.')

                EVEN = (n_steps % rate) == 0
                if not EVEN:
                    zero_tensor = tf.zeros_like(Pinputs[0])
                    dialated_n_steps = n_steps // rate + 1

                    for i_pad in range(dialated_n_steps * rate - n_steps):
                        Pinputs.append(zero_tensor)
                else:
                    dialated_n_steps = n_steps // rate

                dilated_inputs = [tf.concat(Pinputs[i * rate:(i + 1) * rate],axis=0) for i in range(dialated_n_steps)]

                if rate != 1:
                    labels2[0]=tf.concat(labels2[0:rate],axis=0)

                def _loop_function(prev,i):
                    result = prev
                    return result

                scope_name = "Dilation_"+str(rate)
                _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, dilated_inputs, dtype=tf.float32,scope=scope_name)

                dilated_outputs, _ = legacy_seq2seq.rnn_decoder(labels2, enc_state, decoding_cells,loop_function=_loop_function,scope=scope_name)
                splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
                sublist1 = [sublist for sublist in splitted_outputs]
                unrolled_outputs = [output for output in sublist1]
                unrolled_outputs_array=[]
                for i in range(len(unrolled_outputs)):
                    for j in range((len(unrolled_outputs[i]))):
                            unrolled_outputs_array.append(unrolled_outputs[i][j])
                unrolled_outputs_array=unrolled_outputs_array[:horizon]
                outputs = tf.stack(unrolled_outputs_array, axis=1)
                Totaloutputlist.append(outputs)

            Totaloutputlist= tf.reshape(Totaloutputlist, (len(Totaloutputlist),batch_size, horizon, num_nodes, output_dim))
            output_avg= tf.reduce_mean(Totaloutputlist, axis=0)

            self._outputs = output_avg
            self._merged = tf.summary.merge_all()

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
