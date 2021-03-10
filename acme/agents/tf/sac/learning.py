# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAC learner implementation."""

import time
from typing import List

import acme
from acme import types
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl

tfd = tfp.distributions

'''
def SAC_losses(q_t:     tf.Tensor,
               v_t:     tf.Tensor,
               log_prob_t:     tf.Tensor,
               r_t:     tf.Tensor,
               gamma_t: tf.Tensor,
               v_tp1:   tf.Tensor,
               entropy_coeff: float) -> tf.Tensor:
  # Recover linear action from environment action
  squashed_a_t = 2 * (a_t - self._offset)/self._scale - 1 # [-1, 1]
  linear_a_t = tf.atanh(squashed_a_t)
  log_probs = self._distribution.log_prob(samples)
'''

class SACLearner(acme.Learner):
  """SAC learner.

  This is the learning component of a SAC agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      sampling_head: snt.Module,
      discount: float,
      entropy_coeff: float,
      dataset: tf.data.Dataset,
      observation_network: types.TensorTransformation = lambda x: x,
      policy_optimizer: snt.Optimizer = None,
      critic_optimizer: snt.Optimizer = None,
      clipping: bool = False,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
  ):
    """Initializes the learner.

    Args:
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      discount: discount to use for TD updates.
      dataset: dataset to learn from, whether fixed or from a replay buffer
        (see `acme.datasets.reverb.make_dataset` documentation).
      observation_network: an optional online network to process observations
        before the policy and the critic.
      policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
      critic_optimizer: the optimizer to be applied to the critic loss.
      clipping: whether to clip gradients by global norm.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    # Store online and target networks.
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._sampling_head = sampling_head

    # Make sure observation networks are snt.Module's so they have variables.
    self._observation_network = tf2_utils.to_sonnet_module(observation_network)

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

    # Other learner parameters.
    self._discount = discount
    self._clipping = clipping
    self._entropy_coeff = entropy_coeff

    # Create an iterator to go through the dataset.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    # Create optimizers if they aren't given.
    self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
    self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)

    self._checkpointer = tf2_savers.Checkpointer(
        time_delta_minutes=10,
        objects_to_save={
            'counter': self._counter,
            'policy': self._policy_network,
            'critic': self._critic_network,
            'encoder': self._observation_network,
            'policy_optimizer': self._policy_optimizer,
            'critic_optimizer': self._critic_optimizer,
        },
        enable_checkpointing=checkpoint,
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @tf.function
  def _step(self):
    # Update target network.
    online_variables = (
        *self._observation_network.variables,
        *self._critic_network.variables,
        *self._policy_network.variables,
    )

    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

    # Cast the additional discount to match the environment discount dtype.
    discount = tf.cast(self._discount, dtype=d_t.dtype)

    with tf.GradientTape(persistent=True) as tape:
      # Maybe transform the observation before feeding into policy and critic.
      # Transforming the observations this way at the start of the learning
      # step effectively means that the policy and critic share observation
      # network weights.
      o_tm1 = self._observation_network(o_tm1)
      o_t = self._observation_network(o_t)
      o_t = tree.map_structure(tf.stop_gradient, o_t)

      # Policy
      pol_tm1, v_tm1 = self._policy_network(o_tm1)
      pol_t, v_t = self._policy_network(o_t) # already "squozen"
      pol_t = tree.map_structure(tf.stop_gradient, pol_t)
      v_t = tree.map_structure(tf.stop_gradient, v_t)
      #v_t = tree.map_structure(tf.stop_gradient, v_t)

      # DPG loss. If clipping is true use dqda clipping and clip the norm.
      dqda_clipping = 1.0 if self._clipping else None
      onpol_a_tm1 = self._sampling_head(pol_tm1)
      onpol_q_tm1 = self._critic_network(o_tm1, onpol_a_tm1)
      onpol_q_tm1 = tf.squeeze(onpol_q_tm1, axis=-1)  # [B]
      dpg_loss = losses.dpg(
          onpol_q_tm1,
          onpol_a_tm1,
          tape=tape,
          dqda_clipping=dqda_clipping,
          clip_norm=self._clipping)
      dpg_loss = tf.reduce_mean(dpg_loss, axis=0)
      #tf.print('dpg_loss ', tf.shape(dpg_loss))

      # Actor loss. If clipping is true use dqda clipping and clip the norm.
      # TODO: two critic nets, e.g. q1_tm1 and q2_tm1, pick the min as target
      logP_tm1 = self._sampling_head.log_prob(onpol_a_tm1, pol_tm1)
      entropy_loss = self._entropy_coeff * tf.reduce_mean(logP_tm1, axis=0)

      # V(s) loss
      value_target = tf.stop_gradient(onpol_q_tm1 -self._entropy_coeff*logP_tm1)
      #value_loss = tf.nn.compute_average_loss(
      #tf.print('onpol_q_tm1 ', tf.shape(onpol_q_tm1))
      #tf.print('v_tm1 ', tf.shape(v_tm1))
      value_loss = losses.huber(value_target - v_tm1, 1.0)
      #tf.print('value_loss ', tf.shape(value_loss))
      value_loss = tf.reduce_mean(value_loss, axis=0)

      # Critic learning with TD loss
      q_tm1 = self._critic_network(o_tm1, a_tm1)
      q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]

      onpol_a_t = self._sampling_head(pol_t)
      onpol_q_t = self._critic_network(o_t, onpol_a_t)
      onpol_q_t = tf.squeeze(onpol_q_t, axis=-1)  # [B]
      onpol_q_t = tree.map_structure(tf.stop_gradient, onpol_q_t)

      scaled_r_t = self._observation_network.scale_rewards(r_t)
      #tf.print('scaled_r_t ', scaled_r_t)
      critic_target = tf.stop_gradient(r_t + d_t * tf.minimum(v_t, onpol_q_t))
      #critic_target = tf.stop_gradient(scaled_r_t + d_t * 0.5*(v_t + onpol_q_t))
      #critic_loss = #tf.nn.compute_average_loss(
      critic_loss = losses.huber(critic_target - q_tm1, 1.0)
      #tf.print('critic_loss ', tf.shape(critic_loss))
      critic_loss = tf.reduce_mean(critic_loss, axis=0)

      encoder_loss = self._observation_network.compute_loss(o_tm1, r_t)

      policy_loss = value_loss + entropy_loss + dpg_loss + encoder_loss
      #tf.print('policy_loss ', tf.shape(policy_loss))
      #tf.print('critic_loss ', tf.shape(critic_loss))

    # Get trainable variables.
    # Here we train the preprocessing variables along with the policy.
    policy_variables = (
      self._observation_network.trainable_variables +
      self._policy_network.trainable_variables)
    critic_variables = self._critic_network.trainable_variables

    # Compute gradients.
    policy_gradients = tape.gradient(policy_loss, policy_variables)
    critic_gradients = tape.gradient(critic_loss, critic_variables)

    # Delete the tape manually because of the persistent=True flag.
    del tape

    # Maybe clip gradients.
    if self._clipping:
      policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
      critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]

    # Apply gradients.
    self._policy_optimizer.apply(policy_gradients, policy_variables)
    self._critic_optimizer.apply(critic_gradients, critic_variables)
    #tf.print(self._observation_network._obs_mean)
    #tf.print(self._observation_network._obs_scale)
    # Losses to track.
    return {
        'critic_loss': critic_loss,
        'svalue_loss': value_loss,
        'entropy_loss': entropy_loss,
        'dpg_loss': dpg_loss,
        'avg_q': tf.reduce_mean(onpol_q_t, axis=0),
    }

  def step(self):
    # Run the learning step.
    fetches = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    fetches.update(counts)

    # Checkpoint and attempt to write the logs.
    self._checkpointer.save()
    self._logger.write(fetches)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables[name]) for name in names]
