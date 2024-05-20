import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from bandits_helpers import *

participant_ids = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
CONTEXT_SIZE = 2
CMAP = cm.get_cmap('PiYG')
colors = np.linspace(0, 1, 5)

for participant in participant_ids:
    with open('study1_processed_data/{}.pickle'.format(participant), 'rb') as file:         
        data=pickle.load(file)
    performances, actions, contexts = data['performances'], data['actions'], data['contexts']
    performances = np.array(performances)
    actions = np.array(actions)
    contexts = np.array(contexts)
    # tf_environment = tf_py_environment.TFPyEnvironment(ExerciseEnvironment(CONTEXT_SIZE))
    # observation_spec = tensor_spec.BoundedTensorSpec(shape=(CONTEXT_SIZE,), dtype=tf.float32, minimum=0, maximum=1)
    # time_step_spec = ts.time_step_spec(observation_spec)
    # action_spec = tensor_spec.BoundedTensorSpec(dtype=tf.int32, shape=(), minimum=0, maximum=2)

    # agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
    #                                       action_spec=action_spec, alpha=2, )
    
    # step = tf_environment.reset()
    # all_actions = []
    # all_rewards = []
    # for iter in range(len(performances)):
    #     step, tf_environment, agent, reward_received, action = one_iteration(tf_environment, agent, step, train=True, iteration_num=iter, train_reward=performances[iter], train_action=actions[iter], train_context=contexts[iter])
    #     all_actions.append(action)
    #     all_rewards.append(reward_received)

    # all_actions = np.array(all_actions)
    # all_rewards = np.array(all_rewards)

    fig, ax = plt.subplots(figsize=(15, 4))
    legends = [False, False, False, False, False, False, False, False, False]
    colors = np.linspace(0, 1, 5)
    for ii in range(len(performances)-1):
        context = contexts[ii]
        action = actions[ii]
        reward = performances[ii+1]
        if context[0] == 1:
            if not legends[0]:
                ax.plot(ii, 3, 's', color='b', label='bicep curls', markersize=10)
                legends[0] = True
            else:
                ax.plot(ii, 3, 's', color='b', markersize=10)
        else:
            if not legends[1]:
                ax.plot(ii, 3, 's', color='c', label='lateral raises', markersize=10)
                legends[1] = True
            else:
                ax.plot(ii, 3, 's', color='c', markersize=10)
                
        if context[2] == 1:
            if not legends[2]:
                ax.plot(ii, 2, 's', color='lime', label='Good', markersize=10)
                legends[2] = True
            else:
                ax.plot(ii, 2, 's', color='lime', markersize=10)
        else:
            if not legends[3]:
                ax.plot(ii, 2, 's', color='tomato', label='Bad', markersize=10)
                legends[3] = True
            else:
                ax.plot(ii, 2, 's', color='tomato', markersize=10)
        
        if action == 0:
            if not legends[4]:
                ax.plot(ii, 1, 's', color=CMAP(colors[action+1]), label='Firm', markersize=10)
                legends[4] = True
            else:
                ax.plot(ii, 1, 's', color=CMAP(colors[action+1]), markersize=10)
        if action == 1:
            if not legends[5]:
                ax.plot(ii, 1, 's', color='gray', label='Neutral', markersize=10)
                legends[5] = True
            else:
                ax.plot(ii, 1, 's', color='gray', markersize=10)
        if action == 2:
            if not legends[6]:
                ax.plot(ii, 1, 's', color=CMAP(colors[action+1]), label='Encouraging', markersize=10)
                legends[6] = True
            else:
                ax.plot(ii, 1, 's', color=CMAP(colors[action+1]), markersize=10)
       
        if reward == 1:
            if not legends[7]:
                ax.plot(ii, 0, 's', color='g', label='Good Next Rep', markersize=10)
                legends[7] = True
            else:
                ax.plot(ii, 0, 's', color='g', markersize=10)
        else:
            if not legends[8]:
                ax.plot(ii, 0, 's', color='r', label='Bad Next Rep', markersize=10)
                legends[8] = True
            else:
                ax.plot(ii, 0, 's', color='r', markersize=10)
        
    ax.legend()
    plt.yticks([], [])
    ax.set_xlabel('Reps')
    ax.set_xlim([0, len(performances)+15])
    ax.set_title('Participant {}'.format(participant))
    fig.savefig('study1_plots/participant_{}.png'.format(participant))
    

