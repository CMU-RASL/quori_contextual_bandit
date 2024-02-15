import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from tf_agents.bandits.agents import lin_ucb_agent

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os

import warnings
warnings.filterwarnings("ignore") 

#Parameters
NUM_ITERATIONS = 200
CONTEXT_SIZE = 1
INERTIA = 0 #Prob of staying the same style
CMAP = cm.get_cmap('PiYG')

def fatigue_function(x):
    y = np.array([0.001, 0.01, 0.07, 0.15, 0.3, 0.5, 0.65, 0.75, 0.8, 0.8]) - 0.5
    return y[x%10]

class Human:
    def __init__(self, alpha, response_params, fatigue_effect):
        
        self.alpha = alpha
        self.response_params = response_params
        self.fatigue_effect = fatigue_effect

    def get_response(self, action, observation):

        #Updated prob correct based on action
        fatigue = observation + 0.5

        #Response based on fatigue levels and fatigue effect
        self.prob_correct = self.response_params * (1 - (self.alpha * fatigue * self.fatigue_effect)) 
        self.prob_correct = np.minimum(np.array([0.95, 0.95, 0.95, 0.95, 0.95]), np.maximum(np.array([0.05, 0.05, 0.05, 0.05, 0.05]), self.prob_correct))
        if np.random.random() < self.prob_correct[action]:
            return 1
        else:
            return 0
    
    def get_expected_optimal_response(self):
        return np.max(self.prob_correct), np.argmax(self.prob_correct)

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
    
  def __init__(self, alpha, response_params, fatigue_effect, mapping):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(CONTEXT_SIZE,), dtype=np.float32, minimum=-1, maximum=1, name='observation')
    self._current_iteration = 0
    self._human = Human(alpha, response_params, fatigue_effect)
    self._mapping = mapping
    super(ExerciseEnvironment, self).__init__(self._observation_spec, self._action_spec)
  
  def _observe(self):
    self._observation = fatigue_function(self._current_iteration)
    self._current_iteration += 1
    return np.array(self._observation, dtype='float32').reshape((1,))
  
  def _apply_action(self, action):
    return self._human.get_response(self._mapping.index(action), self._observation)
  
  def optimal_reward(self):
     return self._human.get_expected_optimal_response()

def trajectory_for_bandit(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))

if __name__ == '__main__':
  
  alpha_vec = [1]
  human_profiles_vec = ['No Preference Low Performer', 'No Preference High Performer', 'Encouraging Preference', 'Firm Preference', 'Extreme Preference']
  response_params_vec = [np.array([0.5, 0.5, 0.5, 0.5, 0.5]), np.array([0.8, 0.8, 0.8, 0.8, 0.8]), np.array([0.3, 0.4, 0.5, 0.6, 0.8]), np.array([0.8, 0.6, 0.5, 0.4, 0.3]), np.array([0.8, 0.6, 0.4, 0.6, 0.8])]
  fatigue_effect_vec = [np.array([1.0, 1.0, 1.0, 1.0, 1.0]), np.array([-1.5, -0.5, 0.25, 1.25, 2]), np.array([2, 1.25, 0.25, -0.5, -1.5])]
  array = [0, 1, 2, 3, 4]
  mappings = [random.sample( array, len(array) ) for ii in range(10)]

  #True mapping -> false map (map[action])
  #False map -> True map (map.index(action))

  #Initialize environment
  for alpha in alpha_vec:
     for human_profile, response_params in zip(human_profiles_vec, response_params_vec):
        for fatigue_effect in fatigue_effect_vec:

          if (alpha == 0 and np.sum(fatigue_effect) == 5) or alpha > 0:

            all_actions = []
            all_rewards = []
            all_regrets = []

            for map_num, map in enumerate(mappings):
              print(map_num+1, 'out of', len(mappings))
              tf_environment = tf_py_environment.TFPyEnvironment(ExerciseEnvironment(alpha, response_params, fatigue_effect, map))

              observation_spec = tensor_spec.BoundedTensorSpec(shape=(CONTEXT_SIZE,), dtype=tf.float32, minimum=0, maximum=3)
              time_step_spec = ts.time_step_spec(observation_spec)
              action_spec = tensor_spec.BoundedTensorSpec(dtype=tf.int32, shape=(), minimum=0, maximum=4)

              agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                        action_spec=action_spec, alpha=2)

              step = tf_environment.reset()
              observations = []
              actions = []
              rewards = []
              regret = []
              optimal_rewards = []
              optimal_actions = []
              for _ in range(NUM_ITERATIONS):
                action_step = agent.collect_policy.action(step)
                
                next_step = tf_environment.step(action_step.action)
                experience = trajectory_for_bandit(step, action_step, next_step)
                opt_reward, opt_action = tf_environment.pyenv.envs[0].optimal_reward()
                optimal_rewards.append(opt_reward)
                optimal_actions.append(opt_action)
                fatigue = experience.observation.numpy()
                if fatigue < 0.1:
                  binned_fatigue = 0.0
                elif fatigue < 0.5:
                  binned_fatigue = 1.0
                elif fatigue < 0.75:
                  binned_fatigue = 2.0
                else:
                  binned_fatigue = 3.0
                experience.replace(observation=tf.convert_to_tensor(binned_fatigue))
                agent.train(experience)
                step = next_step

                actions.append(int(np.squeeze(np.squeeze(experience.action.numpy()))))
                observations.append(float(np.squeeze(experience.observation.numpy())))
                rewards.append(float(np.squeeze(experience.reward.numpy())))
                regret.append(optimal_rewards[-1] - rewards[-1])

              actions = [map.index(a) for a in actions]
              all_actions.append(actions)
              all_regrets.append(regret)
              all_rewards.append(rewards)

            #Create colorbar plot
            all_actions = np.array(all_actions)
            all_regrets = np.array(all_regrets)
            all_rewards = np.array(all_rewards)
            
            unique, counts = np.unique(all_actions.reshape(-1,1), return_counts=True)
            mapping = dict(zip(unique, counts))
            distributions = []
            for action_val in range(5):
              if action_val in mapping.keys():
                distributions.append(mapping[action_val])
              else:
                distributions.append(0)
            
            distributions = np.array(distributions)
            distributions = distributions / np.sum(distributions)
            print(distributions)

            fig1, ax1 = plt.subplots()
            prev_val = 0
            colors = np.linspace(0, 1, 5)
            action_labels = ['Very Firm', 'Firm', 'Neutral', 'Encouraging', 'Very Encouraging']
            for action_val, dist_val in enumerate(distributions):
              if action_val == 2:
                c = 'gray'
              else:
                c = CMAP(colors[action_val])
              patch = ax1.barh(0, dist_val, left=prev_val, color=c, label=action_labels[action_val])
              p = patch.get_children()[0]
              bl = p.get_xy()
              x = 0.5*p.get_width() + bl[0]
              y = 0.5*p.get_height() + bl[1]
              ax1.text(x,y, "%d%%" % (distributions[action_val]*100), ha='center')
              prev_val = dist_val+prev_val

            ax1.legend()
            ax1.axis([0, 1, -3, 3])
            ax1.set_axis_off()
            average_reward = np.mean(np.sum(all_rewards, axis=1))
            average_regret = np.mean(np.sum(all_regrets, axis=1))
            title = 'Gamma = {}, Fatigue Effect = {} \n Average Reward = {}, Average Regret = {}'.format(alpha, fatigue_effect, np.round(average_reward, 2), np.round(average_regret, 2))
            ax1.set_title(title)
            fig1.savefig('plots/plot_{}_{}_{}.png'.format(human_profile, alpha, fatigue_effect))

            #See temporal changes
            fig2, ax2 = plt.subplots()
            for rep_num in range(10):
              indices = np.arange(rep_num, len(actions)+rep_num, 10).astype('int')
              rep_actions = []
              for ii in range(len(mappings)):
                rep_actions.extend([all_actions[ii, jj] for jj in indices])
              rep_unique, rep_counts = np.unique(rep_actions, return_counts=True)
              rep_mapping = dict(zip(rep_unique, rep_counts))
              rep_distributions = []
              for action_val in range(5):
                if action_val in rep_mapping.keys():
                  rep_distributions.append(rep_mapping[action_val])
                else:
                  rep_distributions.append(0)
              
              rep_distributions = np.array(rep_distributions)
              rep_distributions = rep_distributions / np.sum(rep_distributions)

              prev_val = 0
              colors = np.linspace(0, 1, 5)
              action_labels = ['Very Firm', 'Firm', 'Neutral', 'Encouraging', 'Very Encouraging']
              for action_val, dist_val in enumerate(rep_distributions):
                if action_val == 2:
                  c = 'gray'
                else:
                  c = CMAP(colors[action_val])
                if rep_num == 0:
                  patch = ax2.barh(-rep_num, dist_val, left=prev_val, color=c, label=action_labels[action_val], align='center')
                else:
                  patch = ax2.barh(-rep_num, dist_val, left=prev_val, color=c)
                p = patch.get_children()[0]
                bl = p.get_xy()
                x = 0.5*p.get_width() + bl[0]
                y = 0.5*p.get_height() + bl[1]
                if rep_distributions[action_val] > 0:
                  ax2.text(x,y, "%d%%" % (rep_distributions[action_val]*100), ha='center', va='center', fontsize='small')
                prev_val = dist_val+prev_val
              
            ax2.legend()
            ax2.axis([0, 1, -10, 7])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.get_xaxis().set_ticks([])
            ax2.set_yticks(np.arange(-9, 1), labels=['Rep {}'.format(rep_num) for rep_num in np.arange(10, 0, -1)])
            ax2.set_title(title)
            fig2.savefig('temporal_plots/temporal_plots_{}_{}_{}.png'.format(human_profile, alpha, fatigue_effect))

            #See results split out temporally
            for map_ind in range(len(mappings)):
              fig3, ax3 = plt.subplots()

              action_matrix = np.zeros((10, int(NUM_ITERATIONS/10)))
              start_idx = 0
              for set_num in range(int(NUM_ITERATIONS/10)):
                set_actions = all_actions[map_ind, start_idx:start_idx+10]
                action_matrix[:, set_num] = set_actions*0.2
                start_idx += 10
            
              ax3.imshow(action_matrix, cmap=CMAP)
              ax3.set_xticks(np.arange(int(NUM_ITERATIONS/10)), labels=['Set {}'.format(ii+1) for ii in range(int(NUM_ITERATIONS/10))])
              ax3.set_yticks(np.arange(10), labels=['Rep {}'.format(ii+1) for ii in range(10)])
              plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
              ax3.set_title(title)
              fig3.savefig('all_actions/all_actions_{}_{}_{}_{}.png'.format(human_profile, alpha, fatigue_effect, map_ind))
              
            print('Saved Fig', '{} Gamma {} Fatigue Effect {}\n'.format(human_profile, alpha, fatigue_effect))
            # plt.show()
            plt.close('all')

'''
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
        print('Should have gone to', action_step.action.numpy(), 'actually went to', new_action)
        action_step = action_step.replace(action=tf.convert_to_tensor([new_action]))
'''