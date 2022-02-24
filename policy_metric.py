import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend

class PolicyStat(keras.metrics.Metric):

    def __init__(self, num_act, name='policy_stat', **kwargs):

        super().__init__(name=name, **kwargs)  #, trainable=False

        weight_shape = (num_act,)
        self.per_action_pos = self.add_weight(name='per_action_pos', shape=weight_shape, initializer='zeros')
        self.per_action_neg = self.add_weight(name='per_action_neg', shape=weight_shape, initializer='zeros')
        self.action_pos = self.add_weight(name='action_pos', shape=weight_shape, initializer='zeros')
        self.action_neg = self.add_weight(name='action_neg', shape=weight_shape, initializer='zeros')
        self.action_sum = self.add_weight(name='action_sum', shape=weight_shape, initializer='zeros')

        pass  # __init__()

    def update_state(self, y_true, log_probs, sample_weight=None):

        act_rewards = y_true[:,:3]               # (?,3)
        #log_probs, values, features = model(x)  # (?,3)
        #print_ndarray('act_rewards, log_prob', np.concatenate([act_rewards, log_probs], axis=-1), 16, frm='8.3f')

        # `true` actions
        #true_log_probs = tf.math.log_softmax(act_rewards)   # (?, 3)
        #true_actions = tf.expand_dims(tf.math.argmax(true_log_probs, axis=1, output_type=tf.int32), axis=-1) # (?, 1)
        #true_actions_act = tf.one_hot(true_actions[:,0], 3, dtype=tf.float32)      # (?,3)
        #print_ndarray('act_rewards, true_log_probs', np.concatenate([act_rewards, true_actions, true_actions_act], axis=-1), 16, frm='6.0f')

        #log_probs = tf.math.log_softmax(tf.ones_like(act_rewards))  # (?, 3)
        #policy_actions = tf.random.categorical(log_probs, 1, dtype=tf.int32)
        policy_actions = tf.expand_dims(tf.math.argmax(log_probs, axis=1, output_type=tf.int32), axis=-1)    # (?, 1)
        policy_actions_act = tf.one_hot(policy_actions[:,0], 3, dtype=tf.float32)  # (?,3)
        #print_ndarray('act_rewards, log_probs', np.concatenate([act_rewards, policy_actions, policy_actions_act], axis=-1), 16, frm='6.0f')

        #  tf.math.greater(a, 0.0)
        action_pos_mask = tf.where(act_rewards > 0, 1.0, 0.0)   # (?,3)
        action_neg_mask = tf.where(act_rewards < 0, 1.0, 0.0)   # (?,3)
        #print_ndarray('act_rewards, log_probs', np.concatenate([act_rewards, policy_actions, policy_actions_act, action_pos_mask, action_neg_mask], axis=-1), 16, frm='5.0f')

        per_action_pos = tf.reduce_sum(action_pos_mask * policy_actions_act, axis=0)  # (3,)
        per_action_neg = tf.reduce_sum(action_neg_mask * policy_actions_act, axis=0)  # (3,)
        action_pos = tf.reduce_sum(action_pos_mask, axis=0)     # (3,)
        action_neg = tf.reduce_sum(action_neg_mask, axis=0)     # (3,)
        action_sum = tf.reduce_sum(policy_actions_act, axis=0)  # (3,)

        self.per_action_pos.assign_add(per_action_pos)
        self.per_action_neg.assign_add(per_action_neg)
        self.action_pos.assign_add(action_pos)
        self.action_neg.assign_add(action_neg)
        self.action_sum.assign_add(action_sum)

        pass  # update_state()

    def __str__(self):
        action_sum, per_action_pos, per_action_neg = self.result().numpy()
        per_action_pos = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_pos[1:]]) + ']'
        per_action_neg = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_neg]) + ']'
        action_sum = '[' + ''.join(['{:5.1f}'.format(s) for s in action_sum]) + ']'
        return f'{action_sum} {per_action_pos} {per_action_neg}'

    def result(self):
        epsilon = 1e-6
        action_sum = self.action_sum / tf.reduce_sum(self.action_sum + epsilon) * 100.
        per_action_pos = self.per_action_pos/(self.action_pos + epsilon) * 100.
        per_action_neg = self.per_action_neg/(self.action_sum + epsilon) * 100.
        return (action_sum, per_action_pos, per_action_neg)

    def reset_state(self):
        backend.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])
        pass  # reset_state()

    pass  # PolicyStat


