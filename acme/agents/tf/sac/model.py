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

from typing import Tuple

from acme import types
from acme import specs
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sonnet as snt

tfd = tfp.distributions

class MDPNormalization(snt.Module):

  def __init__(self,
               env_spec: specs.EnvironmentSpec,
               max_ep_length: int = 1000,
               gamma: float = 0.99,
               name: str = 'MDP_normalization_layer'):
    super().__init__(name=name)
    obs_spec = env_spec.observations
    self._obs_mean  = tf.Variable(tf.zeros(
      obs_spec.shape, obs_spec.dtype), name="obs_mean")
    self._obs_scale = tf.Variable(tf.ones(
      obs_spec.shape, obs_spec.dtype), name="obs_scale")
    self._ret_mean  = tf.Variable(tf.zeros(
      1, obs_spec.dtype), name="ret_mean")
    self._ret_scale = tf.Variable(tf.ones(
      1, obs_spec.dtype), name="ret_scale")
    self._ret_factor = 1.0 / min(1.0 * max_ep_length, 1.0/(1.0 - gamma + 1e-16))

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    output = self._obs_scale * (inputs - self._obs_mean)
    output = tf.stop_gradient(output)
    return output

  def scale_rewards(self, rewards: tf.Tensor) -> tf.Tensor:
    output = self._ret_scale * (rewards - self._ret_mean)
    output = tf.stop_gradient(output)
    return output

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
    samples = policy_means + self._distribution.sample()
    #tf.print('samples', samples)
    squashed = tf.math.tanh(samples)  # [-1, 1]
    actions = squashed * self._scale + self._mean  # [minimum, maximum]
    return actions

  def log_prob(self,
               actions: tf.Tensor,
               policy_means: tf.Tensor) -> tf.Tensor:
    squashed = (actions - self._mean) / self._scale # [-1, 1]
    samples = tf.math.atanh(squashed) # linear space, like output of policy net
    log_probs = self._distribution.log_prob(samples - policy_means)
    log_probs -= tf.reduce_sum(tf.math.log(1 - squashed ** 2), axis=-1)
    #tf.print('samples ', tf.shape(samples-policy_means))
    return log_probs

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
    return self._sample(policy)
