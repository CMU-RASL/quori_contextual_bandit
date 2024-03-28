import abc
print('Imported abc')
import numpy as np
print('Imported numpy')
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
import matplotlib.pyplot as plt
print('Imported plt')
import matplotlib.cm as cm
print('Imported cm')
import random
print('Imported random')

import warnings
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
    def __init__(self, alpha, response_params, fatigue_effect, sampling):
      self.alpha = alpha
      self.response_params = response_params
      self.fatigue_effect = fatigue_effect
      self.sampling = sampling

    def calc_rewards_per_action(self, observation, iteration):
      fatigue_mapping = [0.027, 0.2375, 0.46666667]
      fatigue = fatigue_mapping[np.argmax(observation)]
      
      rewards = self.response_params * (1 - (self.alpha * fatigue * self.fatigue_effect)) 
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
    
  def __init__(self, alpha, response_params, fatigue_effect, mapping, sampling):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(CONTEXT_SIZE,), dtype=np.float32, minimum=0, maximum=1, name='observation')
    self._current_iteration = 0
    self._human = Human(alpha, response_params, fatigue_effect, sampling)
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

def run_simulation(map, num_train_iterations, num_test_iterations, tf_environment, agent, optimality, pseudo, action_freq):
  step = tf_environment.reset()
  observations = -1*np.ones((num_train_iterations + num_test_iterations, CONTEXT_SIZE))
  actions = -1*np.ones((num_train_iterations + num_test_iterations))
  rewards = -1*np.ones((num_train_iterations + num_test_iterations))
  optimal_rewards = -1*np.ones((num_train_iterations + num_test_iterations))

  observations_seen = -1*np.ones((num_train_iterations + num_test_iterations, CONTEXT_SIZE))

  for ii in range(num_train_iterations + num_test_iterations):

    #Get the observation
    observation = step.observation.numpy()
    observations[ii, :] = observation

    #If we are at the correct frequency to act
    if ii % action_freq == 0:

      #Set observation to be the mean of observations
      # if ii < action_freq:
      #    observations_seen[ii,:] = observations[ii,:]
      # else:
      #   observations_seen[ii,:] = np.mean(observations[ii-action_freq:ii,:], axis=0)
      observations_seen[ii,:] = observations[ii,:]
      
      #Get optimal action/reward
      # print(observations[ii,:], observations[np.min(0,ii-action_freq):ii,:], observations_seen[ii,:])
      opt_reward, opt_action = tf_environment.pyenv.envs[0].optimal_reward([observations_seen[ii,:]]) #opt_action in true action space
      opt_action = map[opt_action] #now in false action space

      #Choose action
      action_step = agent.collect_policy.action(step)

      #If optimal, replace action with optimal action
      if optimality=='opt':
        action_step = action_step.replace(action=tf.convert_to_tensor([opt_action])) #Replace the action with optimal action (false space)

      next_step = tf_environment.step(action_step.action)

      experience = trajectory_for_bandit(step, action_step, next_step)

      reward_received = next_step.reward.numpy()
      action_chosen_false = int(np.squeeze(np.squeeze(experience.action.numpy())))
      action_chosen_true = map.index(action_chosen_false)

      actions[ii] = action_chosen_true
      rewards[ii] = reward_received
      optimal_rewards[ii] = opt_reward

      agent.train(experience)

    else:
      observations_seen[ii,:] = [-1, -1, -1]

      action_step = agent.collect_policy.action(step)

      next_step = tf_environment.step(action_step.action)

      experience = trajectory_for_bandit(step, action_step, next_step)
    
    step = next_step

  #Return just the test iterations
  test_rewards = rewards[num_train_iterations:]
  test_optimal_rewards = optimal_rewards[num_train_iterations:]
  test_actions = actions[num_train_iterations:]
  test_regrets = test_rewards - test_optimal_rewards

  return test_actions, test_rewards, test_regrets
   
def temporal_plot(actions, title, fig_name):
  fig, ax = plt.subplots(figsize=[7, 10])
  rep_actions = np.zeros((6, 10))
  rep_num = 0
  for ii in range(actions.shape[0]):
    if int(actions[ii]) == -1:
      rep_actions[5, rep_num] += 1
    else:
      rep_actions[int(actions[ii]), rep_num] += 1
    rep_num += 1
    if rep_num == 10:
      rep_num = 0

  rep_distributions = rep_actions / np.sum(rep_actions, axis=0)
  colors = np.linspace(0, 1, 5)
  action_labels = ['Very Firm', 'Firm', 'Neutral', 'Encouraging', 'Very Encouraging', 'No Action']
  colors = np.linspace(0, 1, 5)

  for rep_num in range(10):
    prev_val = 0
    for action_val in range(6):
      if action_val == 2:
        c = 'gray'
      elif action_val == 5:
        c = 'brown'
      else:
        c = CMAP(colors[action_val])
      
      if rep_num == 0:
        patch = ax.barh(-rep_num, rep_distributions[action_val, rep_num], left=prev_val, color=c, label=action_labels[action_val], align='center')
      else:
        patch = ax.barh(-rep_num, rep_distributions[action_val, rep_num], left=prev_val, color=c)
      p = patch.get_children()[0]
      bl = p.get_xy()
      x = 0.5*p.get_width() + bl[0]
      y = 0.5*p.get_height() + bl[1]
      if rep_distributions[action_val, rep_num] > 0:
        ax.text(x,y, "%d%%" % (rep_distributions[action_val, rep_num]*100), ha='center', va='center', fontsize='small')
      prev_val = rep_distributions[action_val, rep_num]+prev_val

  ax.legend()
  ax.axis([0, 1, -10, 7])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.get_xaxis().set_ticks([])
  ax.set_yticks(np.arange(-9, 1), labels=['Rep {}'.format(rep_num) for rep_num in np.arange(10, 0, -1)])
  ax.set_title(title)
  fig.savefig('{}.png'.format(fig_name))

if __name__ == '__main__':
  print('Everything loaded')
  alpha_vec = [1]
  human_profiles_vec = ['Firm Preference']#, 'Encouraging Preference']
  response_params_vec = [np.array([0.8, 0.6, 0.5, 0.4, 0.3])]#, np.array([0.3, 0.4, 0.5, 0.6, 0.8])]
  fatigue_effect_vec = [ np.array([4, 2, 1, -0.5, -2]), np.array([4, 2, 1, -0.5, -2])]
  array = [0, 1, 2, 3, 4]
  mappings = [random.sample( array, len(array) ) for ii in range(5)]

  #True mapping -> false map (map[action])
  #False map -> True map (map.index(action))

  pseudo_reward_mapping = np.array([
    [-1, 0.75, -1, -1, -1],
    [0.75, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0.75],
    [-1, -1, -1, 0.75, -1]
  ])

  #Initialize environment
  for alpha in alpha_vec:
     for human_profile, response_params in zip(human_profiles_vec, response_params_vec):
        for fatigue_effect in fatigue_effect_vec:
            for optimality in ['opt', 'not_opt']:
              for pseudo in ['not_pseudo']:
                for action_freq in [3]:
                  for sampling in [False, True]:
                  
                    all_actions = []
                    all_rewards = []
                    all_regrets = []
                    for map_num, map in enumerate(mappings):
                      
                      #Initialize environment and agent
                      tf_environment = tf_py_environment.TFPyEnvironment(ExerciseEnvironment(alpha, response_params, fatigue_effect, map, sampling))

                      observation_spec = tensor_spec.BoundedTensorSpec(shape=(CONTEXT_SIZE,), dtype=tf.float32, minimum=0, maximum=1)
                      time_step_spec = ts.time_step_spec(observation_spec)
                      action_spec = tensor_spec.BoundedTensorSpec(dtype=tf.int32, shape=(), minimum=0, maximum=4)

                      agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                                action_spec=action_spec, alpha=2, )
                      
                      #Training + Test
                      actions, rewards, regrets  = run_simulation(map, NUM_TRAIN_ITERATIONS, NUM_TEST_ITERATIONS, tf_environment, agent, optimality, pseudo, action_freq)

                      all_actions.append(actions)
                      all_rewards.append(rewards)
                      all_regrets.append(regrets)

                      print('\t Finished mapping {}/{}'.format(map_num+1, len(mappings)))
                    
                    all_actions = np.hstack(all_actions)
                    all_rewards = np.hstack(all_rewards)
                    all_regrets = np.hstack(all_regrets)

                    actions_taken = np.where(all_actions >= 0)[0]
                    total_reward = np.mean(all_rewards[actions_taken])
                    total_regret = np.mean(all_regrets[actions_taken])
                    if sampling:
                      sampling_text = 'sampling'
                    else:
                      sampling_text = 'not sampling'
                    name = 'profile_{}_alpha_{}_fatigue_{}_train_{}_test_{}'.format(human_profile, alpha, fatigue_effect, NUM_TRAIN_SETS, NUM_TEST_SETS)
                    fig_name = 'temporal_plots/{}/{}/{}/action_freq {}/{}'.format(optimality, pseudo, sampling_text, action_freq, name)
                    title = 'Gamma = {}, Fatigue Effect = {} \n Average Reward = {}, Average Regret = {}\n Train Sets = {}, Test Sets = {}\n Optimality = {}, Pseudo = {}\n Action Freq = {}, Sampling = {}'.format(alpha, fatigue_effect, np.round(total_reward, 4), np.round(total_regret, 4), NUM_TRAIN_SETS, NUM_TEST_SETS, optimality, pseudo, action_freq, sampling)


                    #Create temporal plot
                    temporal_plot(all_actions, title, fig_name)
                    print('Generated Plot {}'.format(fig_name))
                    plt.close('all')