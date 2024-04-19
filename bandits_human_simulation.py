import abc
print('Imported abc')
import tensorflow as tf
print('Imported tensorflow')
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore") 

#Parameters
NUM_TRAIN_SETS = 30
NUM_TRAIN_ITERATIONS = 10*NUM_TRAIN_SETS
NUM_TEST_SETS = 20
NUM_TEST_ITERATIONS = 10*NUM_TEST_SETS
CONTEXT_SIZE = 3
CMAP = cm.get_cmap('PiYG')

def fatigue_function(x):
  fatigue1 = np.array([1, 0, 0])
  fatigue2 = np.array([0, 1, 0])
  fatigue3 = np.array([0, 0, 1])
  series = np.stack([fatigue1, fatigue1, fatigue1, fatigue2, fatigue2, fatigue2, fatigue2, fatigue3, fatigue3, fatigue3])
  return series[x%10,:]

class Human:
    def __init__(self, response_params, fatigue_effect, sampling):
      self.response_params = response_params
      self.fatigue_effect = fatigue_effect
      self.sampling = sampling

    def calc_rewards_per_action(self, observation, iteration):
      fatigue_mapping = [0.027, 0.2375, 0.46666667]
      fatigue = fatigue_mapping[np.argmax(observation)]
      
      rewards = self.response_params * (1 - (fatigue * self.fatigue_effect)) 
      rewards = np.minimum(np.array([0.95, 0.95, 0.95, 0.95, 0.95]), np.maximum(np.array([0.05, 0.05, 0.05, 0.05, 0.05]), rewards))

      if fatigue == 0.2375:
        rewards[-1] = 0.3
      return rewards

    def get_response(self, action, observation, iteration):
      #action is in true space
      rewards = self.calc_rewards_per_action(observation, iteration)
      if self.sampling:
        if np.random.random() < rewards[action]:
          return 1
        else:
          return 0
      else:
        return rewards[action]
    
    def get_expected_optimal_response(self, observation, iteration):
      rewards = self.calc_rewards_per_action(observation, iteration)
      return np.max(rewards), np.argmax(rewards)

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
    
  def __init__(self, response_params, fatigue_effect, mapping, sampling):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(CONTEXT_SIZE,), dtype=np.float32, minimum=0, maximum=1, name='observation')
    self._current_iteration = 0
    self._human = Human(response_params, fatigue_effect, sampling)
    self._mapping = mapping
    super(ExerciseEnvironment, self).__init__(self._observation_spec, self._action_spec)
  
  def _observe(self):
    self._observation = fatigue_function(self._current_iteration)
    self._current_iteration += 1
    return np.array(self._observation, dtype='float32').reshape((CONTEXT_SIZE,))
  
  def _apply_action(self, action):
    #Action is in false mapping
    true_action = self._mapping.index(action)
    return self._human.get_response(true_action, self._observation, self._current_iteration)
  
  def optimal_reward(self, observation):
     return self._human.get_expected_optimal_response(observation, self._current_iteration)

def trajectory_for_bandit(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))

def one_iteration(tf_environment, agent, optimality, map, action_freq, step, train, iteration_num, previous_action_true):
  observation = step.observation.numpy()

  if iteration_num % action_freq == 0:

    opt_reward, opt_action_true = tf_environment.pyenv.envs[0].optimal_reward([observation]) #opt_action in true action space
    opt_action_false = map[opt_action_true] #now in false action space

    #Choose action
    action_step = agent.collect_policy.action(step)

    #If optimal, replace action with optimal action
    if optimality=='opt':
      action_step = action_step.replace(action=tf.convert_to_tensor([opt_action_false])) #Replace the action with optimal action (false space)

    next_step = tf_environment.step(action_step.action)

    experience = trajectory_for_bandit(step, action_step, next_step)

    reward_received = next_step.reward.numpy()
    action_chosen_false = int(np.squeeze(np.squeeze(experience.action.numpy())))
    action_chosen_true = map.index(action_chosen_false)
    regret = opt_reward - reward_received

    if train:
      agent.train(experience)
  else:
    if previous_action_true > -1:
      opt_reward, opt_action_true = tf_environment.pyenv.envs[0].optimal_reward([observation]) #opt_action in true action space
      opt_action_false = map[opt_action_true] #now in false action space

      action_step = agent.collect_policy.action(step)
      previous_action_false = map[previous_action_true]
      action_step = action_step.replace(action=tf.convert_to_tensor([previous_action_false]))
      next_step = tf_environment.step(action_step.action)
      experience = trajectory_for_bandit(step, action_step, next_step)

      reward_received = next_step.reward.numpy()
      action_chosen_false = int(np.squeeze(np.squeeze(experience.action.numpy())))
      action_chosen_true = map.index(action_chosen_false)
      regret = opt_reward - reward_received

      if train:
        agent.train(experience)

    else:
      action_step = agent.collect_policy.action(step)
      next_step = tf_environment.step(action_step.action)
      experience = trajectory_for_bandit(step, action_step, next_step)
    
      reward_received = -1e100
      regret = -1e100
      action_chosen_true = -1e100
      opt_action_true = -1e100

  step = next_step

  return step, tf_environment, agent, reward_received, regret, action_chosen_true, opt_action_true

def process_data(rewards, regrets, actions, opt_actions, response_param, fatigue_effect, optimality, action_freq, sampling, train_set_vec):

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
    title = 'Response Param = {}, Fatigue Effect = {}, Average Reward = {:.4f}, Average Regret = {:.4f} \n Train Sets = {}, Test Sets = {}, Optimality = {}, Action Freq = {}, Sampling = {}'.format(response_param, fatigue_effect, cur_rewards, cur_regrets, train_set_num, 20, optimality, action_freq, sampling)
    fig.suptitle(title)
    directory = 'plots/sampling_{}/action_freq_{}/train_sets_{}'.format(sampling, action_freq, train_set_num)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig_name = '{}/response_param_{}_fatigue_effect_{}'.format(directory, response_param, fatigue_effect)
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
  fig.savefig('plots/performance_results.png')

if __name__ == '__main__':
  print('Everything loaded')
  response_params_vec = [np.array([0.8, 0.6, 0.5, 0.4, 0.3])]#, np.array([0.3, 0.4, 0.5, 0.6, 0.8])]
  fatigue_effect_vec = [np.array([4, 2, 1, -0.5, -2])] ##, np.array([4, 2, 1, -0.5, -2])]
  array = [0, 1, 2, 3, 4]
  mappings = [random.sample( array, len(array) ) for ii in range(5)]

  #True mapping -> false map (map[action])
  #False map -> True map (map.index(action))

  #Want to plot overall performance
  train_set_vec = [10, 20, 30, 40, 50, 60, 70, 80]
  test_sets = 20
  results = pd.DataFrame(columns=['response_param', 'fatigue_effect', 'optimality', 'action_freq', 'sampling', 'train_set_num', 'reward', 'regret'])

  for response_param in response_params_vec:
    for fatigue_effect in fatigue_effect_vec:
      for optimality in ['not_opt']:
        for action_freq in [1, 2, 3]:
          for sampling in [False, True]:
            print('Starting {} {} {} {}'.format(response_param, fatigue_effect, optimality, action_freq))

            all_rewards = np.zeros((len(mappings), len(train_set_vec), test_sets, 10))
            all_regrets = np.zeros((len(mappings), len(train_set_vec), test_sets, 10))
            all_actions = np.zeros((len(mappings), len(train_set_vec), test_sets, 10))
            all_optimal_actions = np.zeros((len(mappings), len(train_set_vec), test_sets, 10))

            for map_num, map in enumerate(mappings):
              print('\tStarting map {} out of {}'.format(map_num+1, len(mappings)))
              #Initialize environment and agent
              tf_environment = tf_py_environment.TFPyEnvironment(ExerciseEnvironment(response_param, fatigue_effect, map, sampling))

              observation_spec = tensor_spec.BoundedTensorSpec(shape=(CONTEXT_SIZE,), dtype=tf.float32, minimum=0, maximum=1)
              time_step_spec = ts.time_step_spec(observation_spec)
              action_spec = tensor_spec.BoundedTensorSpec(dtype=tf.int32, shape=(), minimum=0, maximum=4)

              agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                        action_spec=action_spec, alpha=2, )
              
              step = tf_environment.reset()
              sets_trained = 0
              train_iteration = 0
              train_previous_action_true = -1
              for train_set_ind in range(len(train_set_vec)):
                while sets_trained < train_set_vec[train_set_ind]:
                  #10 iterations per set
                  for iter in range(10):
                    step, tf_environment, agent, reward, regret, action, opt_action = one_iteration(tf_environment, agent, optimality, map, action_freq, step, train=True, iteration_num=train_iteration, previous_action_true=train_previous_action_true)
                    train_iteration += 1
                    train_previous_action_true = action
                  sets_trained += 1
                  if sets_trained % 5 == 0:
                    print('\t\tTrained {} sets'.format(sets_trained))

                #Test for 20 sets
                test_iteration = 0
                test_previous_action_true = -1
                for test_set_ind in range(test_sets):
                  for iter in range(10):
                    step, tf_environment, agent, reward, regret, action, opt_action = one_iteration(tf_environment, agent, optimality, map, action_freq, step, train=False, iteration_num=test_iteration, previous_action_true=test_previous_action_true)
                    all_rewards[map_num, train_set_ind, test_set_ind, iter] = reward
                    all_regrets[map_num, train_set_ind, test_set_ind, iter] = regret
                    all_actions[map_num, train_set_ind, test_set_ind, iter] = action
                    all_optimal_actions[map_num, train_set_ind, test_set_ind, iter] = opt_action
                    test_iteration += 1
                    test_previous_action_true = action
                  if (test_set_ind + 1) % 5 == 0:
                    print('\t\t\tTested {} sets'.format(test_set_ind+1))

            
            saved_reward, saved_regret = process_data(all_rewards, all_regrets, all_actions, all_optimal_actions, response_param, fatigue_effect, optimality, action_freq, sampling, train_set_vec)

            for train_set_num, r, s in zip(train_set_vec, saved_reward, saved_regret):
              results.loc[-1] = [response_param, fatigue_effect, optimality, action_freq, sampling, train_set_num, r, s]
              results.index = results.index + 1
              results = results.sort_index()

          
  plot_performance_results(results)