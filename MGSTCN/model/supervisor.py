import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml

from lib import utils, metrics
from lib.metrics import masked_mae_loss
from model.model import MGSTCNModel


class MGSTCNSupervisor(object):

    def __init__(self, adj_mx, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._log_dir = self._get_log_dir(kwargs)

        # Data preparation
        self._data = utils.load_dataset(**self._data_kwargs)

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('MGSTCN', reuse=False):
                self._train_model = MGSTCNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('MGSTCN', reuse=True):
                self._test_model = MGSTCNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=adj_mx, **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)

        preds = self._train_model.outputs
        labels = self._train_model.labels

        null_val = 0.
        self._loss_fn = masked_mae_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 10)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        print('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            k = kwargs['model'].get('k')
            horizon = kwargs['model'].get('horizon')
            run_id = 'mgstcn_%d_h_%d_lr_%g_bs_%d/' % (k, horizon, learning_rate, batch_size)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mae': loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            maes.append(vals['mae'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })
    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50,epochs=100,min_learning_rate = 2e-6, lr_decay_ratio = 0.1,  save_model=1, **train_kwargs):

        history = []
        min_val_loss = float('inf')

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        print('Start training ...')

        while self._epoch <= epochs:
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True)
            train_loss, train_mae = train_results['loss'], train_results['mae']


            global_step = sess.run(tf.train.get_or_create_global_step())
            val_results = self.run_epoch_generator(sess, self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])

            utils.add_simple_summary(['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                                     [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, global_step, train_mae, val_mae, new_lr, (end_time - start_time))


            print(message)

            if val_loss <= min_val_loss:
                wait = 0
                self.evaluate(sess)
                if save_model > 0:
                    self.save(sess, val_loss)
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    print('Early stopping at epoch: %d' % self._epoch)
                    break
            history.append(val_mae)
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self, sess):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                return_output=True,
                                                training=False)

        test_loss, y_preds = test_results['loss'], test_results['outputs']
        utils.add_simple_summary(['loss/test_loss'], [test_loss], global_step=global_step)

        y_preds = np.concatenate(y_preds, axis=0)
        scaler = self._data['scaler']
        predictions = []
        y_truths = []
        for horizon_i in [2,5,11]:
            y_truth = scaler.inverse_transform(self._data['y_test'][:, horizon_i:(horizon_i+1), :, 0])
            y_truths.append(y_truth)
            y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], horizon_i:(horizon_i+1), :, 0])
            predictions.append(y_pred)

            mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
            mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
            print( "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(horizon_i + 1, mae, mape, rmse))
            utils.add_simple_summary(['%s_%d' % (item, horizon_i + 1) for item in
                                      ['metric/rmse', 'metric/mape', 'metric/mae']],
                                     [rmse, mape, mae],global_step=global_step)
        outputs = {'predictions': predictions, 'groundtruth': y_truths}
        return outputs

    def load(self, sess, model_filename):
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']

