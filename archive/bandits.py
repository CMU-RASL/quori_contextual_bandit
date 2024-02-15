
# import abc
import numpy as np
import tensorflow as tf

# from tf_agents.agents import tf_agent
from tf_agents.drivers import driver, dynamic_step_driver
# from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
# from tf_agents.policies import tf_policy
# from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
# from tf_agents.trajectories import trajectory
# from tf_agents.trajectories import policy_step

from tf_agents.metrics import tf_metrics as tf_agent_metrics

from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import matplotlib.pyplot as plt
import numpy as np

nest = tf.nest

batch_size = 1 # @param
context_size = 1 # @param

#Generates the context randomly - can modify this to pull from a list later
def context_sampling_fn(batch_size):
  """Contexts from [-10, 10]^context_size."""
  def _context_sampling_fn():
    return np.random.randint(-10, 10, [batch_size, context_size]).astype(np.float32)
  return _context_sampling_fn

class LinearNormalReward(object): #reward is linear function of the context + some noise
  """A class that acts as linear reward function when called."""
  def __init__(self, theta, sigma):
    self.theta = theta
    self.sigma = sigma
  def __call__(self, x):
    mu = np.dot(x, np.ones_like(self.theta)) #np.dot(x, self.theta)
    return np.random.normal(mu, self.sigma)

# arm0_param = [-3, 0, 1, -2] # @param 
# arm1_param = [1, -2, 3, 0] # @param
# arm2_param = [0, 0, 1, 1] # @param


#Let's try arm2 as highest, then arm0, then arm1 
arm0_param = context_size*[1] # @param firm
arm1_param = context_size*[2] # @param neutral
arm2_param = context_size*[5] # @param encouraging

arm0_reward_fn = LinearNormalReward(arm0_param, 1)
arm1_reward_fn = LinearNormalReward(arm1_param, 1)
arm2_reward_fn = LinearNormalReward(arm2_param, 1)

environment = tf_py_environment.TFPyEnvironment(
    sspe.StationaryStochasticPyEnvironment(
        context_sampling_fn(batch_size),
        [arm0_reward_fn, arm1_reward_fn, arm2_reward_fn],
        batch_size=batch_size))

observation_spec = tensor_spec.TensorSpec([context_size], tf.float32)
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = tensor_spec.BoundedTensorSpec(
    dtype=tf.int32, shape=(), minimum=0, maximum=2)

agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                     action_spec=action_spec)

def compute_optimal_reward(observation):
  expected_reward_for_arms = [
      tf.linalg.matvec(observation, tf.cast(arm0_param, dtype=tf.float32)),
      tf.linalg.matvec(observation, tf.cast(arm1_param, dtype=tf.float32)),
      tf.linalg.matvec(observation, tf.cast(arm2_param, dtype=tf.float32))]
  optimal_action_reward = tf.reduce_max(expected_reward_for_arms, axis=0)
  print(observation, expected_reward_for_arms, optimal_action_reward)
  return optimal_action_reward

regret_metric = tf_metrics.RegretMetric(compute_optimal_reward)

num_iterations = 5 # @param
steps_per_loop = 1 # @param

chosen_action_metric = tf_agent_metrics.ChosenActionHistogram(buffer_size=num_iterations)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.policy.trajectory_spec,
    batch_size=batch_size,
    max_length=steps_per_loop)

observers = [replay_buffer.add_batch, regret_metric, chosen_action_metric]

driver = dynamic_step_driver.DynamicStepDriver(
    env=environment,
    policy=agent.collect_policy,
    num_steps=steps_per_loop * batch_size,
    observers=observers)

regret_values = []

for iter in range(num_iterations):
  if iter % 10 == 0:
    print(iter)
  driver.run()
  loss_info = agent.train(replay_buffer.gather_all())
  replay_buffer.clear()
  regret_values.append(regret_metric.result())
  actions = chosen_action_metric.result()
  # print(actions)

# print(actions)
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(regret_values)
ax[0].set_ylabel('Average Regret')
ax[0].set_xlabel('Number of Iterations')

ax[1].plot(actions.numpy())
ax[1].set_ylabel('Action Chosen')
ax[1].set_xlabel('Number of Iterations')
plt.show()