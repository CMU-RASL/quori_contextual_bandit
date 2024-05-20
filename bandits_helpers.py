import abc
print('Imported abc')
from tf_agents.environments import py_environment
print('Imported py_environment')
from tf_agents.environments import tf_py_environment
print('Imported tf_py_environment')
from tf_agents.specs import array_spec
print('Imported array_spec')
from tf_agents.specs import tensor_spec
print('Imported tensor_spec')
from tf_agents.trajectories import time_step as ts
print('Imported time_step')
from tf_agents.trajectories import trajectory
print('Imported trajectory')
from tf_agents.bandits.agents import lin_ucb_agent
print('Imported lin_ucb_agent')
import tensorflow as tf
print('Imported tensorflow')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os
import warnings
import pandas as pd
import datetime
warnings.filterwarnings("ignore") 

#Parameters
CMAP = cm.get_cmap('PiYG')

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
    
  def __init__(self, context_size):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(context_size,), dtype=np.float32, minimum=0, maximum=1, name='observation')
    self._current_iteration = 0
    self._context_size = context_size
    super(ExerciseEnvironment, self).__init__(self._observation_spec, self._action_spec)
  
  def _observe(self):
    return np.array([0, 0], dtype='float32').reshape((self._context_size,))
  
  def _apply_action(self, action):
    return -1

def trajectory_for_bandit(initial_step, action_step, final_step):
    return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))

def one_iteration(tf_environment, agent, step, train, iteration_num, train_reward, train_action, train_context):
    
    step = step._replace(observation=tf.convert_to_tensor([train_context], dtype=tf.float32))

    #Choose action           
    action_step = agent.collect_policy.action(step)

    #Replace action with train_action
    action_step = action_step.replace(action=tf.convert_to_tensor([train_action]))

    next_step = tf_environment.step(action_step.action)
    next_step = next_step._replace(reward=tf.convert_to_tensor([float(train_reward)]))

    experience = trajectory_for_bandit(step, action_step, next_step)
    
    action_chosen = int(np.squeeze(np.squeeze(experience.action.numpy())))
    
    reward_received = float(experience.reward.numpy())

    if train:
        agent.train(experience)
        match action_chosen:
            case 0: #very firm
                experience.replace(action=tf.convert_to_tensor([1]))
                agent.train(experience)
                pass
            case 1: #firm
                experience.replace(action=tf.convert_to_tensor([0]))
                agent.train(experience)
                pass
            case 2: #neutral
                pass
            case 3: #encouraging
                experience.replace(action=tf.convert_to_tensor([4]))
                agent.train(experience)
                pass
            case 4: #very encouraging
                experience.replace(action=tf.convert_to_tensor([3]))
                agent.train(experience)
                pass

    step = next_step
    return step, tf_environment, agent, reward_received, action_chosen

def process_data(rewards, regrets, actions, opt_actions, response_param, fatigue_effect, optimality, action_freq, sampling, pseudo, train_set_vec):

  saved_rewards = []
  saved_regrets = []
  #For each train_set
  for train_set_ind, train_set_num in enumerate(train_set_vec):
    cur_rewards = rewards[:, train_set_ind, :, :]
    cur_regrets = regrets[:, train_set_ind, :, :]
    cur_actions = actions[:, train_set_ind, :, :]
    cur_opt_actions = opt_actions[:, train_set_ind, :, :]

    #Average over mappings   
    cur_rewards = np.mean(cur_rewards, where=cur_rewards> -100)
    cur_regrets = np.mean(cur_regrets, where=cur_regrets> -100)
    
    #Action distribution
    actions_chosen = np.zeros((6, 10))
    opt_actions_chosen = np.zeros((6, 10))
    for iter in range(10):
      for action in range(6):
        if action == 5:
          actions_chosen[action, iter] = len(np.where(cur_actions[:, :, iter] < -100)[0])
          opt_actions_chosen[action, iter] = len(np.where(cur_opt_actions[:, :, iter] < -100)[0])
        else:
          actions_chosen[action, iter] = len(np.where(cur_actions[:, :, iter] == action)[0])
          opt_actions_chosen[action, iter] = len(np.where(cur_opt_actions[:, :, iter] == action)[0])
    fig, ax = plt.subplots(1, 2, figsize=[14, 8])
    action_dist = actions_chosen / np.sum(actions_chosen, axis=0)
    opt_action_dist = opt_actions_chosen / np.sum(opt_actions_chosen, axis=0)
    colors = np.linspace(0, 1, 5)
    action_labels = ['Very Firm', 'Firm', 'Neutral', 'Encouraging', 'Very Encouraging', 'No Action']

    for rep_num in range(10):
      prev_val0 = 0
      prev_val1 = 0
      for action_val in range(6):
        if action_val == 2:
          c = 'gray'
        elif action_val == 5:
          c = 'brown'
        else:
          c = CMAP(colors[action_val])
        
        if rep_num == 0:
          patch0 = ax[0].barh(-rep_num, action_dist[action_val, rep_num], left=prev_val0, color=c, label=action_labels[action_val], align='center')
          patch1 = ax[1].barh(-rep_num, opt_action_dist[action_val, rep_num], left=prev_val1, color=c, label=action_labels[action_val], align='center')
        else:
          patch0 = ax[0].barh(-rep_num, action_dist[action_val, rep_num], left=prev_val0, color=c)
          patch1 = ax[1].barh(-rep_num, opt_action_dist[action_val, rep_num], left=prev_val1, color=c)

        p = patch0.get_children()[0]
        bl = p.get_xy()
        x = 0.5*p.get_width() + bl[0]
        y = 0.5*p.get_height() + bl[1]
        if action_dist[action_val, rep_num] > 0:
          ax[0].text(x,y, "%d%%" % (action_dist[action_val, rep_num]*100), ha='center', va='center', fontsize='small')
        prev_val0 = action_dist[action_val, rep_num]+prev_val0

        p = patch1.get_children()[0]
        bl = p.get_xy()
        x = 0.5*p.get_width() + bl[0]
        y = 0.5*p.get_height() + bl[1]
        if opt_action_dist[action_val, rep_num] > 0:
          ax[1].text(x,y, "%d%%" % (opt_action_dist[action_val, rep_num]*100), ha='center', va='center', fontsize='small')
        prev_val1 = opt_action_dist[action_val, rep_num]+prev_val1

      for ind in range(2):
        ax[ind].axis([0, 1, -10, 7])
        ax[ind].spines['top'].set_visible(False)
        ax[ind].spines['right'].set_visible(False)
        ax[ind].spines['bottom'].set_visible(False)
        ax[ind].spines['left'].set_visible(False)
        ax[ind].get_xaxis().set_ticks([])
        ax[ind].set_yticks(np.arange(-9, 1), labels=['Rep {}'.format(rep_num) for rep_num in np.arange(10, 0, -1)])
      
    ax[1].legend()
    ax[0].set_title('Actual Actions')
    ax[1].set_title('Optimal Actions')
    title = 'Response Param = {}, Fatigue Effect = {}, Average Reward = {:.4f}, Average Regret = {:.4f} \n Train Sets = {}, Test Sets = {}, Optimality = {}, Action Freq = {}, Sampling = {}, Pseudo = {}'.format(response_param, fatigue_effect, cur_rewards, cur_regrets, train_set_num, 20, optimality, action_freq, sampling, pseudo)
    fig.suptitle(title)
    directory = 'plots/sampling_{}/action_freq_{}/train_sets_{}'.format(sampling, action_freq, train_set_num)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig_name = '{}/response_param_{}_fatigue_effect_{}_pseudo_{}'.format(directory, response_param, fatigue_effect, pseudo)
    fig.savefig('{}.png'.format(fig_name))

    saved_rewards.append(cur_rewards)
    saved_regrets.append(cur_regrets)
  
  return saved_rewards, saved_regrets

def plot_performance_results(data):
  fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[12, 8])
  #Sampling False on left, Sampling True on right
  #Regret first row, Reward second row
  colors = {1: 'red', 2: 'blue', 3: 'olive'}
  for col, sampling in zip([0, 1], [False, True]):
    for row, metric in zip([0, 1], ['regret', 'reward']):
      for action_freq in [1, 2, 3]:
        d = data.loc[(data['sampling'] == sampling) & (data['action_freq'] == action_freq), ['train_set_num', metric]]
        if len(d) > 0:
          ax[row, col].plot(d['train_set_num'], d[metric], '-o', label='Action Freq = {}'.format(action_freq), linewidth=2, markersize=8, color=colors[action_freq])
      ax[row, col].set_title('{} - Sampling = {}'.format(metric, sampling))
      ax[row, col].set_xlabel('Train Set Number')
      ax[row, col].legend()
  fig.savefig('plots/performance_results{}.png'.format(datetime.datetime.now().strftime('%Y%m%d%H%M')))