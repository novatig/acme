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

"""DDPG agent implementation."""

import copy

from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.sac import learning
from acme.agents.tf.sac import model
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import numpy as np
import sonnet as snt
import tensorflow as tf

class SAC(agent.Agent):
  """SAC Agent.

  This implements a single-process SAC agent. This is an actor-critic algorithm
  that generates data via a behavior policy, inserts N-step transitions into
  a replay buffer, and periodically updates the policy (and as a result the
  behavior) by sampling uniformly from this buffer.
  """

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy_network: snt.Module,
               critic_network: snt.Module,
               encoder_network: types.TensorTransformation = tf.identity,
               entropy_coeff: float = 0.01,
               target_update_period: int = 0,
               discount: float = 0.99,
               batch_size: int = 256,
               policy_learn_rate: float = 3e-4,
               critic_learn_rate: float = 5e-4,
               prefetch_size: int = 4,
               min_replay_size: int = 1000,
               max_replay_size: int = 250000,
               samples_per_insert: float = 64.0,
               n_step: int = 5,
               sigma: float = 0.5,
               clipping: bool = True,
               logger: loggers.Logger = None,
               counter: counting.Counter = None,
               checkpoint: bool = True,
               replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      observation_network: optional network to transform the observations before
        they are fed into any network.
      discount: discount to use for TD updates.
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      min_replay_size: minimum replay size before updating.
      max_replay_size: maximum replay size.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      n_step: number of steps to squash into a single transition.
      sigma: standard deviation of zero-mean, Gaussian exploration noise.
      clipping: whether to clip gradients by global norm.
      logger: logger object to be used by learner.
      counter: counter object used to keep track of steps.
      checkpoint: boolean indicating whether to checkpoint the learner.
      replay_table_name: string indicating what name to give the replay table.
    """
    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.

    dim_actions = np.prod(environment_spec.actions.shape, dtype=int)
    extra_spec = {  'logP': tf.ones(shape=(1), dtype=tf.float32),
                  'policy': tf.ones(shape=(1, dim_actions), dtype=tf.float32)}
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)

    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(
          environment_spec, extras_spec=extra_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        priority_fns={replay_table_name: lambda x: 1.},
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    # Make sure observation network is a Sonnet Module.
    observation_network = model.MDPNormalization(environment_spec, encoder_network)

    # Get observation and action specs.
    act_spec = environment_spec.actions
    obs_spec = environment_spec.observations

    # Create the behavior policy.
    sampling_head = model.SquashedGaussianSamplingHead(act_spec, sigma)
    self._behavior_network = model.PolicyValueBehaviorNet(
      snt.Sequential([observation_network, policy_network]), sampling_head)

    # Create variables.
    emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])
    tf2_utils.create_variables(policy_network, [emb_spec])
    tf2_utils.create_variables(critic_network, [emb_spec, act_spec])

    # Create the actor which defines how we take actions.
    actor = model.SACFeedForwardActor(self._behavior_network, adder)

    if target_update_period > 0:
      target_policy_network = copy.deepcopy(policy_network)
      target_critic_network = copy.deepcopy(critic_network)
      target_observation_network = copy.deepcopy(observation_network)

      tf2_utils.create_variables(target_policy_network, [emb_spec])
      tf2_utils.create_variables(target_critic_network, [emb_spec, act_spec])
      tf2_utils.create_variables(target_observation_network, [obs_spec])
    else:
      target_policy_network = policy_network
      target_critic_network = critic_network
      target_observation_network = observation_network

    # Create optimizers.
    policy_optimizer = snt.optimizers.Adam(learning_rate=policy_learn_rate)
    critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learn_rate)

    # The learner updates the parameters (and initializes them).
    learner = learning.SACLearner(
        policy_network=policy_network,
        critic_network=critic_network,
        sampling_head=sampling_head,
        observation_network=observation_network,
        target_policy_network=target_policy_network,
        target_critic_network=target_critic_network,
        target_observation_network=target_observation_network,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        target_update_period=target_update_period,
        learning_rate=policy_learn_rate,
        clipping=clipping,
        entropy_coeff=entropy_coeff,
        discount=discount,
        dataset=dataset,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint,
    )

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)

  @property
  def behavior_network(self):
    return self._behavior_network