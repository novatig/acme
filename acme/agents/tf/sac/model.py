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

from typing import Tuple, Optional

from acme import types
from acme import core
from acme import adders
from acme import specs
from acme.tf import variable_utils as tf2_variable_utils
from acme.tf import utils as tf2_utils

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import sonnet as snt
import dm_env

tfd = tfp.distributions

class MDPNormalization(snt.Module):

  def __init__(self,
               env_spec: specs.EnvironmentSpec,
               encoder: types.TensorTransformation,
               name: str = 'MDP_normalization_layer'):
    super().__init__(name=name)
    obs_spec = env_spec.observations
    self._obs_mean  = tf.Variable(tf.zeros(
      obs_spec.shape, obs_spec.dtype), name="obs_mean")
    self._obs_scale = tf.Variable(tf.ones(
      obs_spec.shape, obs_spec.dtype), name="obs_scale")
    self._ret_mean  = tf.Variable(tf.zeros(
      1, obs_spec.dtype), name="ret_mean")
    self._ret_scale = tf.Variable(0.1 * tf.ones(
      1, obs_spec.dtype), name="ret_scale")
    self._encoder = tf2_utils.to_sonnet_module(encoder)

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    output = self._obs_scale * (inputs - self._obs_mean)
    output = tf.stop_gradient(output)
    return output

  def scale_rewards(self, rewards: tf.Tensor) -> tf.Tensor:
    output = self._ret_scale * (rewards - self._ret_mean)
    output = tf.stop_gradient(output)
    return self._encoder(output)

  def compute_loss(self, obs: tf.Tensor, rew: tf.Tensor) -> tf.Tensor:
    obs_loss = 0.5 * ((obs - self._obs_mean) * self._obs_scale) ** 2 \
               - tf.math.log(self._obs_scale)
    rew_loss = 0.5 * ((rew - self._ret_mean) * self._ret_scale) ** 2 \
               - tf.math.log(self._ret_scale)
    loss = tf.reduce_sum(tf.reduce_mean(obs_loss, axis=0), axis=None)
    loss += tf.reduce_mean(rew_loss, axis=0)
    return loss

class SquashedGaussianValueHead(snt.Module):
  """A network with linear value layer and a scaled tanh policy layer.

  The policy is meant to be used as mean of a Squashed Multivariate Gaussian.
  It is squashed to ensure numerical safety of the Jacobian in the probability.
  E.g. If squash_max=4, the mean ranges between -4 to 4, and the action between
  -/+ tanh(4) > 0.999. The max Jacobian is 1/(1-tanh^2(4)) and has sufficient
  precision when represented in float32.
  """

  def __init__(self,
               dim_actions: int,
               squash_max: float = 4.0,
               init_scale: float = 1e-4,
               name: str = 'policy_value_network'):
    super().__init__(name=name)
    self._policy_layer = snt.Linear(dim_actions,
        w_init=tf.initializers.VarianceScaling(init_scale))
    self._value_layer = snt.Linear(1,
        w_init=tf.initializers.VarianceScaling(init_scale))
    self._squash_max = squash_max

  def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    #policy = self._policy_layer(inputs)
    policy = self._squash_max * tf.math.tanh(
        self._policy_layer(inputs) / self._squash_max)
    value = tf.squeeze(self._value_layer(inputs), axis=-1)
    return policy, value

class SquashedGaussianSamplingHead(snt.Module):
  """Sonnet which takes the output of the policy network (the mean) and
  samples actions, squashed and scaled to the agent's specifications.

  It also defines the method log_prob, used to compute the policy loss for
  bounded action spaces as described in Haarnoja, T, et al. "Soft actor-critic
  algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
  """

  def __init__(self,
               spec: specs.BoundedArray,
               stddev: float,
               name: str = 'sample_to_spec'):
    super().__init__(name=name)
    num_dimensions = np.prod(spec.shape, dtype=int)
    self._scale = (spec.maximum - spec.minimum) / 2.0
    self._mean  = (spec.minimum + spec.minimum) / 2.0
    self._distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros((1, num_dimensions)),
        scale_diag=tf.ones((1, num_dimensions)))

  def __call__(self, policy_means: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(policy_means)
    samples = self._distribution.sample()
    squashed = tf.math.tanh(policy_means + samples)  # [-1, 1]
    actions = squashed * self._scale + self._mean  # [minimum, maximum]
    log_probs = self._distribution.log_prob(samples)
    log_probs -= tf.reduce_sum(tf.math.log(1 - squashed ** 2), axis=-1)
    return actions, log_probs

  def log_prob(self,
               actions: tf.Tensor,
               policy_means: tf.Tensor) -> tf.Tensor:
    squashed = (actions - self._mean) / self._scale # [-1, 1]
    samples = tf.math.atanh(squashed) # linear space, like output of policy net
    log_probs = self._distribution.log_prob(samples - policy_means)
    log_probs -= tf.reduce_sum(tf.math.log(1 - squashed ** 2), axis=-1)
    #tf.print('samples ', tf.shape(samples-policy_means))
    return log_probs

class ReFER_module(snt.Module):
  def __init__(self,
               range_max: float = 4.0,
               offpol_tol: float = 0.1,
               name: str = 'ReFER_module'):
    super().__init__(name=name)
    #self._frac_off_pol = tf.Variable( # adaptive
    #                tf.zeros(shape=()), name="frac_off_pol")
    self._beta   = tf.Variable( # adaptive
                    tf.ones(shape=()) * 0.5, name="beta")
    self._range  = (tf.ones(shape=()) / range_max,
                    tf.ones(shape=()) * range_max) # hyper-param
    self._tol    =  tf.ones(shape=()) * offpol_tol # hyper-param

  def loss(self,
           behavior_logP_tm1: tf.Tensor,
           logP_tm1: tf.Tensor) -> tf.Tensor:
    rhos = tf.math.exp(logP_tm1 - behavior_logP_tm1)
    rhos = tf.stop_gradient(rhos)

    n_off_pol = tf.reduce_sum(
      tf.cast(rhos < self._range[0], rhos.dtype) +
      tf.cast(rhos > self._range[1], rhos.dtype))
    frac_off_pol = n_off_pol / tf.cast(tf.shape(logP_tm1)[0], rhos.dtype)

    # equivalent to fix point iteration
    def fix_point_iter_loss(val: tf.Tensor, goToZero: tf.Tensor):
      goToZero = tf.stop_gradient(goToZero)
      if goToZero: return 0.5 * (0.0 - val) ** 2
      else:        return 0.5 * (1.0 - val) ** 2

    # Update estimate of the fraction of far off-policy samples in the RM
    # i.e. decrease the estimate if the current minibatch had fewer far off-
    # policy samples and vice-versa. We use fix_point_iter to keep it in [0,1]
    #frac_off_pol_loss = 0.5 * (self._frac_off_pol - frac_off_pol) ** 2
    # Update penalization coefficient as in the paper
    beta_loss   = fix_point_iter_loss(self._beta, frac_off_pol>self._tol)

    return beta_loss

  def DKL_coef(self) -> tf.Tensor:
    #for compactness and incapsulation, as if we divided entire loss by beta
    #also, we add some max to guard against beta outside of [0, 1]
    coef = tf.maximum((1.0 - self._beta), 0.0) / tf.maximum(self._beta, 1e-4)
    coef = tf.stop_gradient(coef)
    return coef

class PolicyValueBehaviorNet(snt.Module):

  def __init__(self,
               policy_network: snt.Module,
               sampling_head: snt.Module,
               name: str = 'policy_value_behavior'):
    super().__init__(name=name)
    self._policy = policy_network
    self._sample = sampling_head

  def __call__(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    policy, _ = self._policy(inputs)
    actions, _ = self._sample(policy)
    return actions

  def getAll(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    policy, _ = self._policy(inputs)
    actions, logP = self._sample(policy)
    return actions, policy, logP

class SACFeedForwardActor(core.Actor):
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      adder: adders.Adder,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_network = policy_network

    self._prev_logP = None
    self._prev_means = None

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    action, policy, log_prob = self._policy_network.getAll(batched_observation)

    self._prev_logP = log_prob
    self._prev_means = policy

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    extras = {'logP': self._prev_logP,
              'policy': self._prev_means}
    extras = tf2_utils.to_numpy_squeeze(extras)
    self._adder.add(action, next_timestep, extras)

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)
