import numpy as np
import pickle
import os

ids = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
conditions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
exercise_order = ['bicep_curls', 'lateral_raises']
robot_orders = [['1', '2', '3'], ['1', '3', '2']]

phrases = ["Please walk over", "from 1 to 10?", "Rest", "Start lateral raises now", "Get ready for", "Almost done", "Start bicep curls now"]

for participant_id, condition in zip(ids, conditions):
    robot_order = robot_orders[condition]
    total_sets = 0
    performances = []
    actions = []
    contexts = []
    for round_num, robot in zip(['1', '2', '3'], robot_order):
        for exercise in exercise_order:
            for set_num in ['1', '2']:
                log_filename = 'study1_data/Participant_{}_Round_{}_Robot_{}_Exercise_{}_Set_{}.log'.format(participant_id, round_num, robot, exercise, set_num)
                with open(log_filename) as f:
                    f = f.readlines()
                rep_num = 0
                started_set = 0
                action = 1
                for line in f:
                    if 'STARTING SET' in line:
                        started_set += 1
                    if "'speed':" in line and started_set == 1:
                        ind = line.index('evaluation')
                        evaluation = line[ind+14:-3]
                        evaluation = np.fromstring(evaluation, dtype=int, sep=',')
                        # print('Eval', eval)
                        if np.min(evaluation) >= 0:
                            performances.append(1)
                        else:
                            performances.append(0)
                        actions.append(action)

                        if exercise == 'bicep_curls':
                            context1 = [1, 0]
                        else:
                            context1 = [0, 1]
                        
                        if performances[-1] == 1:
                            context2 = [1, 0]
                        else:
                            context2 = [0, 1]
                        context1.extend(context2)
                        contexts.append(context1)

                        rep_num+=1
                    if "Robot says:" in line and started_set == 1:
                        feedback_flag = True
                        for phrase in phrases:
                            if phrase in line:
                                feedback_flag = False
                        if feedback_flag:
                            if robot == '1':
                                action = 1 #neutral
                            elif robot == '2':
                                action = 0 #firm
                            elif robot == '3':
                                action = 2 #encouraging
                total_sets += 1
                print('\tSets', total_sets, 'Reps', rep_num)
    for action in range(3):
        print('\tAction {}: {}'.format(action, np.sum(np.array(actions) == action)))
    
    data = {'performances': performances, 'actions': actions, 'contexts': contexts}
    filename = 'study1_processed_data/{}.pickle'.format(participant_id)
    try:
        os.remove(filename)
    except OSError:
        pass
    dbfile = open(filename, 'ab')
    pickle.dump(data, dbfile)                    
    dbfile.close()
    print(actions)
    print('Created', filename, '\n')