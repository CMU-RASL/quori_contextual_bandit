import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_bandit_agent
from tf_agents.bandits.agents import neural_boltzmann_agent
from tf_agents.bandits.agents import exp3_mixture_agent
from tf_agents.bandits.agents import greedy_multi_objective_neural_agent

import matplotlib.pyplot as plt

nest = tf.nest

CONTEXT_SIZE = 1
NUM_ITERATIONS = 500
ACTION_MEANS = [-2, -1, 0, 1, 5]
INERTIA = 0.75 #probability of staying at same style
PROB_OF_REWARD = 0.75
SIGMA = 1

ACTION_STDS = [SIGMA, SIGMA, SIGMA, SIGMA, SIGMA]

class BanditPyEnvironment(py_environment.PyEnvironment):

  def __init__(self, observation_spec, action_spec):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    super(BanditPyEnvironment, self).__init__()

  # Helper functions.
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _empty_observation(self):
    return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype),
                                 self.observation_spec())

  # These two functions below should not be overridden by subclasses.
  def _reset(self):
    """Returns a time step containing an observation."""
    return ts.restart(self._observe(), batch_size=self.batch_size)

  def _step(self, action):
    """Returns a time step containing the reward for the action taken."""
    reward = self._apply_action(action)
    return ts.termination(self._observe(), reward)

  # These two functions below are to be implemented in subclasses.
  @abc.abstractmethod
  def _observe(self):
    """Returns an observation."""

  @abc.abstractmethod
  def _apply_action(self, action):
    """Applies `action` to the Environment and returns the corresponding reward.
    """

class ExerciseEnvironment(BanditPyEnvironment):
    
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(CONTEXT_SIZE,), dtype=np.float32, minimum=-10, maximum=10, name='observation')
    self._episode_num = 0
    super(ExerciseEnvironment, self).__init__(self._observation_spec, self._action_spec)
  
  def _observe(self):
    self._observation = -10 + (20 / NUM_ITERATIONS)*self._episode_num + 10*np.sin(self._episode_num*0.1)
    self._episode_num += 1
    return np.array(self._observation, dtype='float32').reshape((1,))
    # return np.random.randint(-10, 10, [1, CONTEXT_SIZE]).astype(np.float32)
  
  def _apply_action(self, action):
    if np.random.random() < PROB_OF_REWARD:
      return np.random.normal(ACTION_MEANS[action]*self._observation, ACTION_STDS[action])
    else:
      return 0

tf_environment = tf_py_environment.TFPyEnvironment(ExerciseEnvironment())

observation_spec = tensor_spec.TensorSpec(shape=(CONTEXT_SIZE,), dtype=tf.float32)
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = tensor_spec.BoundedTensorSpec(
    dtype=tf.int32, shape=(), minimum=0, maximum=4)

agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                     action_spec=action_spec, alpha=4)
# agent = linear_bandit_agent.LinearBanditAgent(time_step_spec=time_step_spec,
#                                      action_spec=action_spec)
# agent = neural_boltzmann_agent.NeuralBoltzmannAgent(time_step_spec=time_step_spec,
#                                      action_spec=action_spec)
# agent = exp3_mixture_agent.Exp3MixtureAgent(time_step_spec=time_step_spec,
#                                      action_spec=action_spec)
# agent = greedy_multi_objective_neural_agent.GreedyMultiObjectiveNeuralAgent(time_step_spec=time_step_spec,
#                                      action_spec=action_spec)

def trajectory_for_bandit(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))


def compute_optimal_reward(observation):
  expected_reward_for_arms = []

  for arm in range(len(ACTION_MEANS)):
    expected_reward_for_arms.append(ACTION_MEANS[arm]*observation*PROB_OF_REWARD)

  optimal_action_reward = np.max(expected_reward_for_arms)

  return optimal_action_reward

step = tf_environment.reset()
observations = []
actions = []
rewards = []
optimal_rewards = []
regret = []
for _ in range(NUM_ITERATIONS):
  action_step = agent.collect_policy.action(step)
  if len(actions) > 0:
    if np.squeeze(action_step.action.numpy()) == actions[-1] or INERTIA == 0: # if same action as last time, continue
      pass
    else:
      
      #Change based on how far apart the two styles are
      diff = np.squeeze(action_step.action.numpy()) - actions[-1]
      
      sample = np.random.random()
      if diff < 0:
        vals = range(diff, 0)
      else:
        vals = range(diff, 0, -1)
      
      new_action = False
      for ii in vals:
          prob_switching = INERTIA/(np.abs(ii))
          # print('ii', ii, 'Prob_switching', prob_switching)
          if sample < prob_switching:
            pass
          else:
            new_action = actions[-1] + ii

      if new_action:
        action_step = action_step.replace(action=tf.convert_to_tensor([new_action]))

  next_step = tf_environment.step(action_step.action)
  experience = trajectory_for_bandit(step, action_step, next_step)
  agent.train(experience)
  step = next_step

  actions.append(int(np.squeeze(np.squeeze(experience.action.numpy()))))
  observations.append(float(np.squeeze(experience.observation.numpy())))
  rewards.append(float(np.squeeze(experience.reward.numpy())))
  optimal_rewards.append(np.squeeze(compute_optimal_reward(experience.observation)))
  regret.append(optimal_rewards[-1] - rewards[-1])

print(np.unique(actions, return_counts=True))
# print('-')
# print('actions', actions)
# print('-')
# print('rewards', rewards)
# print('-')
# print('regret', regret)
# print('-')
# print('observation', observations)
# print('-')
# print('optimal rewards', optimal_rewards)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(regret, '-o')
ax[0].set_ylabel('Regret')
ax[0].set_title('Inertia = {}, Prob of reward = {}, Reward Std = {}'.format(INERTIA, PROB_OF_REWARD, SIGMA))

ax[1].plot(actions, '-o')
ax[1].set_ylabel('Action Chosen')
ax[1].set_ylim([0,4])
# ax[1].set_xlabel('Number of Iterations')

ax[2].plot(rewards, '-o')
ax[2].set_ylabel('Rewards')
# ax[2].set_xlabel('Number of Iterations')

ax[3].plot(observations, '-o')
ax[3].set_ylabel('Observations')
ax[3].set_xlabel('Number of Iterations')
plt.show()

    