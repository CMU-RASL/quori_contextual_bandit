import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print('Loaded tensorflow')


from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.agents import lin_ucb_agent


from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.trajectories import trajectory
print('Loaded tf agents packages')

nest = tf.nest

batch_size = 1 # @param
context_size = 1
arm0_param = [-2]*context_size # @param
arm1_param = [-1]*context_size # @param
arm2_param = [0]*context_size # @param
arm3_param = [-1]*context_size # @param
arm4_param = [2]*context_size # @param
num_iterations = 10 # @param
steps_per_loop = 1 # @param

inertia = 0.75 #probability of staying at same style
p = 1
sigma = 0

def context_sampling_fn(batch_size):
  def _context_sampling_fn():
    obs = np.random.randint(-10, 10, [batch_size, context_size]).astype(np.float32)
    return obs
  return _context_sampling_fn

class LinearNormalReward(object):
  """A class that acts as linear reward function when called."""
  def __init__(self, theta, p, sigma):
    self.theta = theta
    self.p = p
    self.sigma = sigma
  def __call__(self, x):
    mu = np.dot(x, self.theta)
    if np.random.random() < self.p:
      return np.random.normal(mu, self.sigma)
    else:
      return 0

arm0_reward_fn = LinearNormalReward(arm0_param, p, sigma)
arm1_reward_fn = LinearNormalReward(arm1_param, p, sigma)
arm2_reward_fn = LinearNormalReward(arm2_param, p, sigma)
arm3_reward_fn = LinearNormalReward(arm3_param, p, sigma)
arm4_reward_fn = LinearNormalReward(arm4_param, p, sigma)

environment = tf_py_environment.TFPyEnvironment(
    sspe.StationaryStochasticPyEnvironment(
        context_sampling_fn(batch_size),
        [arm0_reward_fn, arm1_reward_fn, arm2_reward_fn, arm3_reward_fn, arm4_reward_fn],
        batch_size=batch_size))


observation_spec = tensor_spec.TensorSpec([context_size], tf.float32)
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = tensor_spec.BoundedTensorSpec(
    dtype=tf.int32, shape=(), minimum=0, maximum=4)

agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
                                     action_spec=action_spec)

def compute_optimal_reward(observation):
  expected_reward_for_arms = [
      arm0_reward_fn.p* tf.linalg.matvec(observation, tf.cast(arm0_reward_fn.theta, dtype=tf.float32)),
      arm1_reward_fn.p* tf.linalg.matvec(observation, tf.cast(arm1_reward_fn.theta, dtype=tf.float32)),
      arm2_reward_fn.p* tf.linalg.matvec(observation, tf.cast(arm2_reward_fn.theta, dtype=tf.float32)),
      arm3_reward_fn.p* tf.linalg.matvec(observation, tf.cast(arm3_reward_fn.theta, dtype=tf.float32)),
      arm4_reward_fn.p* tf.linalg.matvec(observation, tf.cast(arm4_reward_fn.theta, dtype=tf.float32))]
  optimal_action_reward = tf.reduce_max(expected_reward_for_arms, axis=0)
  return optimal_action_reward

def trajectory_for_bandit(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))


step = environment.reset()
observations = []
actions = []
rewards = []
optimal_rewards = []
regret = []
for iter in range(num_iterations):
  action_step = agent.collect_policy.action(step)
  if len(actions) > 0:
    if np.squeeze(action_step.action.numpy()) == actions[-1] or inertia == 0: # if same action as last time, continue
      pass
    else:
      # #Change based on how far apart the two styles are
      # diff = np.abs(np.squeeze(action_step.action.numpy()) - actions[-1])
      
      # #The further away, the less likely to change
      # if np.random.random() < inertia/diff:
      #   action_step = action_step.replace(action=tf.convert_to_tensor([actions[-1]]))
      
      #Change based on how far apart the two styles are
      diff = np.squeeze(action_step.action.numpy()) - actions[-1]
      
      sample = np.random.random()
      if diff < 0:
        vals = range(diff, 0)
      else:
        vals = range(diff, 0, -1)
      
      # print('Considering ', vals, 'Diff', diff, 'Sample', sample)
      new_action = False
      for ii in vals:
          prob_switching = inertia/(np.abs(ii))
          # print('ii', ii, 'Prob_switching', prob_switching)
          if sample < prob_switching:
            pass
          else:
            new_action = actions[-1] + ii

      # print('Previous Action', actions[-1], 'Proposed Action', action_step.action.numpy(), 'New Action', new_action)
      # print('-')
      if new_action:
        action_step = action_step.replace(action=tf.convert_to_tensor([new_action]))

  next_step = environment.step(action_step.action)
  experience = trajectory_for_bandit(step, action_step, next_step)
  print(step, action_step, next_step)
  agent.train(experience)

  actions.append(np.squeeze(experience.action.numpy()))
  observations.append(np.squeeze(experience.observation.numpy()))
  rewards.append(np.squeeze(experience.reward.numpy()))
  optimal_rewards.append(np.squeeze(compute_optimal_reward(experience.observation.numpy()).numpy()))
  regret.append(optimal_rewards[-1] - rewards[-1])
  # print('Iteration', iter)
  # print('Observation:', observations[-1], 'Action:', actions[-1], 'Reward:', rewards[-1], 'Optimal Reward:', optimal_rewards[-1], 'Regret', regret[-1])
  step = next_step

print(np.unique(actions, return_counts=True))

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(regret)
ax[0].set_ylabel('Regret')
ax[0].set_title('Inertia = {}, P = {}, Sigma = {}'.format(inertia, p, sigma))

ax[1].plot(actions, '-o')
ax[1].set_ylabel('Action Chosen')
ax[1].set_ylim([0,4])
# ax[1].set_xlabel('Number of Iterations')

ax[2].plot(rewards)
ax[2].set_ylabel('Rewards')
# ax[2].set_xlabel('Number of Iterations')

ax[3].plot(observations)
ax[3].set_ylabel('Observations')
ax[3].set_xlabel('Number of Iterations')
plt.show()