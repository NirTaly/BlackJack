import math
import os
import itertools
import simulator as sym
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import optuna
# from joblib import parallel_backend
import sqlite3
import matplotlib
matplotlib.use('Agg')

action_dict = {0: 'S', 1: 'H', 2: 'X', 3: 'D', 4: 'P'}
MILLION = 1000000

#TODO add bust-after-double
#TODO add bust-after-hit
#TODO gamma=1                                                       [V]
#TODO start_epsilon,end_epsilon,decay_rate to optuna                [V]
#TODO first learn HARD,SOFT , then SPLIT                            [V]

def CreateQTable(player_state_tuple, legal_actions):
    Q_table = dict()
    start, stop, step = player_state_tuple
    tuple_player_dealer_list = list(itertools.product(list(range(start, stop, step)), list(range(2, 12))))
    for key in tuple_player_dealer_list:
        Q_table[key] = dict.fromkeys(legal_actions, 0)

    return Q_table

def initBasicStrategy(rules):
    if rules == 1:
        return initBasicStrategy1()
    else:
        pass

def initBasicStrategy1():
    Q_table_hard = CreateQTable((4, 22, 1), ['S', 'H', 'X', 'D'])
    Q_table_soft = CreateQTable((13, 22, 1), ['S', 'H', 'X', 'D'])
    Q_table_split = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])

    # -- HARD --

    # 4-8 against 2-10 *HIT*
    for p in range(4, 12):
        for d in range(2, 12):
            Q_table_hard[(p, d)]['H'] = 1

    # 9 against 3-6 *DOUBLE*
    for d in range(3, 7):
        Q_table_hard[(9, d)]['D'] = 2

    # 10,11 against 2-9 *DOUBLE*
    for p in range(10, 12):
        for d in range(2, 10):
            Q_table_hard[(p, d)]['D'] = 2

    # 11 against 10 *DOUBLE*
    Q_table_hard[(11, 10)]['D'] = 2

    # 10,11 against 10,A *HIT*
    Q_table_hard[(10, 10)]['H'] = 1
    Q_table_hard[(10, 11)]['H'] = 1
    Q_table_hard[(11, 11)]['H'] = 1

    # 12 against 2,3,7,8,9,10 *HIT*
    for d in [2, 3, 7, 8, 9, 10, 11]:
        Q_table_hard[(12, d)]['H'] = 1

    # 12 against 4-6 *STAND*
    for d in range(4, 7):
        Q_table_hard[(12, d)]['S'] = 1

    # 13-16 against 2-6 *STAND*
    for p in range(13, 17):
        for d in range(2, 7):
            Q_table_hard[(p, d)]['S'] = 1

    # 13 against 7-10 *HIT*
    for p in range(12, 17):
        for d in range(7, 12):
            Q_table_hard[(p, d)]['H'] = 1

    # 14,15 against 10,A *SURRENDER*
    for d in range(9, 12):
        Q_table_hard[(16, d)]['X'] = 2
    Q_table_hard[(15, 10)]['X'] = 2

    # 17 against 2-10 *STAND*
    for d in range(2, 12):
        Q_table_hard[(17, d)]['S'] = 1

    # 18+ against ALL *STAND*
    for p in range(18, 22):
        for d in range(2, 12):
            Q_table_hard[(p, d)]['S'] = 1

    # hard_policy = CreatePolicyTable(Q_table_hard)
    # print(pd.DataFrame(hard_policy[4:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(4, 22))))

    # -- SOFT --
    # 13,14 against 2-4,7-11 *HIT*
    for p in range(13, 18):
        for d in range(2, 12):
            Q_table_soft[(p, d)]['H'] = 1

    # 13,14 against 5,6 *DOUBLE*
    for p in range(13, 15):
        for d in range(5, 7):
            Q_table_soft[(p, d)]['D'] = 2

    # 15,16 against 4,5,6 *DOUBLE*
    for p in range(15, 17):
        for d in range(4, 7):
            Q_table_soft[(p, d)]['D'] = 2

    # 17,18 against 3-6 *DOUBLE*
    for p in range(17, 19):
        for d in range(3, 7):
            Q_table_soft[(p, d)]['D'] = 2

    # 18 against 2,7,8 *STAND*
    for d in range(2, 9):
        Q_table_soft[(18, d)]['S'] = 1

    # 18 against 9-11 *HIT*
    for d in range(9, 12):
        Q_table_soft[(18, d)]['H'] = 1

    # 19-21 against ALL *STAND*
    for p in range(19, 22):
        for d in range(2, 12):
            Q_table_soft[(p, d)]['S'] = 1

    # -- SPLIT --

    # 2,3 against 2,3,8,9,10,A *HIT*
    for p in range(2, 4):
        for d in [2, 3, 8, 9, 10, 11]:
            Q_table_split[(p, d)]['H'] = 1

    # 2,3 against 4-7 *SPLIT*
    for p in range(2, 4):
        for d in range(4, 8):
            Q_table_split[(p, d)]['P'] = 1

    # correct 3 against A *SURRENDER*
    Q_table_split[(3, 11)]['X'] = 1

    # 4 against ALL *HIT*
    for d in range(2, 12):
        Q_table_split[(4, d)]['H'] = 1

    # 5 against 2-9 *DOUBLE*
    for d in range(2, 10):
        Q_table_split[(5, d)]['D'] = 2

    # 5 against 10,11 *HIT*
    for d in range(2, 12):
        Q_table_split[(5, d)]['H'] = 1

    # 6 against 2,7,8,9,10 *HIT*
    for d in [2, 7, 8, 9, 10, 11]:
        Q_table_split[(6, d)]['H'] = 1

    # 6 against 3,4,5,6 *SPLIT*
    for d in [3, 4, 5, 6]:
        Q_table_split[(6, d)]['P'] = 1

    # 7 against 8-A *HIT*
    for d in [8, 9, 10, 11]:
        Q_table_split[(7, d)]['H'] = 1

    # 7 against 2,3,4,5,6,7 *SPLIT*
    for d in [2, 3, 4, 5, 6, 7]:
        Q_table_split[(7, d)]['P'] = 1

    # 8 against 2,3,4,5,6,7,8,9 *SPLIT*
    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        Q_table_split[(8, d)]['P'] = 1

    # 9 against 2,3,4,5,6,8,9 *SPLIT*
    for d in [2, 3, 4, 5, 6, 8, 9]:
        Q_table_split[(9, d)]['P'] = 1

    # 9 against A *STAND*
    for d in [7, 10, 11]:
        Q_table_split[(9, d)]['S'] = 1

    # 10 against ALL *STAND*
    for d in range(2, 12):
        Q_table_split[(10, d)]['S'] = 1

    # A against 2-10 *SPLIT*
    for d in range(2, 12):
        Q_table_split[(11, d)]['P'] = 1

    return [Q_table_hard, Q_table_soft, Q_table_split]


class QAgent:

    def __init__(self, alpha, gamma, epsilon, final_epsilon, decay_rate, basicStrategy=False):
        # The BJ Simulator
        self.Game = sym.Game()
        self.basicStrategy = basicStrategy

        self.countDict = dict.fromkeys(range(-50,50), 0)

        if not basicStrategy:
            # Init the table with q_values: HARD(18X10) X SOFT(9X10) X SPLIT(8X10) X ACTION(5)
            # 'SD' = stand after double - pseudo state so don't mix stand-after-hit with stand-after-double
            Q_table_hard = CreateQTable((4, 22, 1), ['S', 'H', 'X', 'D'])
            Q_table_soft = CreateQTable((13, 22, 1), ['S', 'H', 'X', 'D'])
            Q_table_split = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])
            Q_table_SD = CreateQTable((6, 22, 1), ['SD'])
            self.Q_table = [Q_table_hard, Q_table_soft, Q_table_split, Q_table_SD]
        else:
            self.Q_table = initBasicStrategy(1)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = decay_rate
    def _playerStateFromGameState(self, game_state, player_state):
        if game_state == 2:
            if self.Game.playerCards[0][0] == 1:
                player_state = 11
            else:
                player_state //= 2
        return player_state

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.decay_rate
        else:
            self.epsilon = self.final_epsilon

    # function that handle case of early BJ
    def handleBJ(self, explore):
        reward, done = 0, False
        player_state, game_state = self.Game.sum_hands(self.Game.playerCards[0])
        dealer_sum, _ = self.Game.sum_hands(self.Game.dealerCards)
        if player_state == 21 or dealer_sum == 21:
            reward = self.Game.rewardHandler(dealer_sum, [player_state])
            reward = 1.5 * reward if (reward > 0) else reward
            done = True
        return reward, done

    def splitUpdateParamsFirst(self, player_state, first_hand_next_game_state, first_hand_next_player_state):
        game_state, action = 2, 'P'
        sec_hand_player_state, sec_hand_game_state = self.Game.sum_hands(self.Game.playerCards[1])

        player_state = self._playerStateFromGameState(game_state, player_state)

        first_hand_params = (game_state, player_state, action, first_hand_next_game_state, first_hand_next_player_state)
        sec_hand_params = (game_state, player_state, action, sec_hand_game_state, sec_hand_player_state)
        hands_params = [first_hand_params, sec_hand_params]

        old_value = self.Q_table[game_state][(player_state, self.Game.dealer_state)][action]
        reward = 0
        next_max = 0
        for i in range(2):
            game_state, player_state, action, next_game_state, next_player_state = hands_params[i]
            next_max += max(itertools.islice(
                self.Q_table[next_game_state][(next_player_state, self.Game.dealer_state)].values(), 2))

        new_value = self.alpha * (reward + self.gamma * next_max - old_value)

        self.Q_table[game_state][(player_state, self.Game.dealer_state)][action] += new_value

    def get_action(self, game_state, player_state, dealer_state, explore=True):
        if (random.uniform(0, 1) < self.epsilon) and explore:
            # Random Action - Exploring
            if self.Game.first_move:
                if game_state == 2:
                    return action_dict[random.randint(0, 4)]
                else:
                    return action_dict[random.randint(0, 3)]
            else:
                return action_dict[random.randint(0, 1)]
        else:
            if self.Game.first_move:
                player_state = self._playerStateFromGameState(game_state, player_state)
                return max(self.Q_table[game_state][(player_state, dealer_state)],
                           key=self.Q_table[game_state][(player_state, dealer_state)].get)

            else:
                return max(dict(itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)),
                           key=dict(
                               itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)).get)

    def update_parameters(self, game_state, player_state, dealer_state, action, reward, next_game_state,
                          next_player_state):
        player_state = self._playerStateFromGameState(game_state, player_state)

        # Q-Learn formula
        old_value = self.Q_table[game_state][(player_state, dealer_state)][action]
        next_max = 0.0
        double_reward = reward
        if next_player_state > 21 or action in {'S', 'X'}:  # terminal state
            next_max = 0.0
        elif action == 'D':  # next valid states is only 'S'
            reward = 0
            next_max = self.Q_table[3][(next_player_state, dealer_state)]['SD']
        else:
            next_max = max(
                itertools.islice(self.Q_table[next_game_state][(next_player_state, dealer_state)].values(), 2))

        new_value = self.alpha * (reward + self.gamma * next_max - old_value)

        # Update the Q_table
        self.Q_table[game_state][(player_state, dealer_state)][action] += new_value
        if next_player_state <= 21 and action == 'D':
            old_value_double = next_max
            new_value_double = self.alpha * (double_reward - old_value_double)
            self.Q_table[3][(next_player_state, dealer_state)]['SD'] += new_value_double

    def run_loop(self, explore, onlyPairs=False):
        game_state, player_state = self.Game.reset_hands(onlyPairs)
        count = self.Game.get_count()
        reward, done = self.handleBJ(explore)
        last_split_hands_parameters = []
        for i, _ in enumerate(self.Game.playerCards):
            while not done:
                action = self.get_action(game_state, player_state, self.Game.dealer_state, explore)

                next_game_state, next_player_state, reward, done = self.Game.step(action)

                if explore:
                    if action == "P":
                        if self.Game.playerCards[0][0] == 1 and self.Game.splitAcesAndDone:
                            next_player_state, next_game_state = self.Game.sum_hands(self.Game.playerCards[1])
                            last_split_hands_parameters.append(
                                [game_state, player_state, action, next_game_state, next_player_state])
                            break
                        else:
                            self.splitUpdateParamsFirst(player_state, next_game_state, next_player_state)
                    elif done and self.Game.isSplit:
                        last_split_hands_parameters.append(
                            [game_state, player_state, action, next_game_state, next_player_state])
                    else:
                        self.update_parameters(game_state, player_state, self.Game.dealer_state, action, reward,
                                                next_game_state, next_player_state)

                game_state = next_game_state
                player_state = next_player_state

                if onlyPairs:
                    return 0, 0
            if self.Game.isSplit:
                if self.Game.playerCards[0][0] == 1 and self.Game.splitAcesAndDone:
                    break
                player_state, game_state = self.Game.sum_hands(self.Game.playerCards[self.Game.currHand])
                done = False
            else:
                rewards = reward

        if self.Game.isSplit:
            if explore:
                for i in range(2):
                    game_state, player_state, action, next_game_state, next_player_state = last_split_hands_parameters[
                        i]
                    self.update_parameters(game_state, player_state, self.Game.dealer_state, action, reward[i],
                                            next_game_state, next_player_state)
            rewards = sum(reward)
            wins = sum(1 for w in reward if w > 0)
        else:
            wins = 1 if (reward > 0) else 0

        self.countDict[count] += rewards
        return rewards, wins

    def train(self, n_train, onlyPairs=False):
        for i in tqdm(range(1, n_train+1)):
            self.update_epsilon()
            self.run_loop(True,onlyPairs)

    def test(self, n_test):
        total_wins = 0
        hands = 0
        rewards = 0
        self.Game.shoe.rebuild()
        objective_list = []
        for _ in tqdm(range(1, n_test+1)):
            reward, wins = self.run_loop(False)
            rewards += reward
            total_wins += wins
            hands += 1 + self.Game.isSplit
            objective_list.append(rewards)
            if self.Game.isSplit:
                objective_list.append(rewards)
        return total_wins / hands, rewards, objective_list, range(1,hands+1)


def validation(gamma):
    """
    best gamma = 0.001
    best alpha = 0.08
    """
    n_learning = 10000000
    alphas = np.arange(0.001, 0.01, 0.001)
    # alphas = [0.005]

    best_alpha = alphas[0]
    best_reward = -math.inf
    for alpha in alphas:
        agent = QAgent(alpha, gamma, epsilon=0.6)
        rewards = 0
        for i in tqdm(range(0, (9 * n_learning) // 10)):
            agent.update_epsilon()
            agent.run_loop(True)
        for _ in tqdm(range(0, n_learning // 10)):
            rewards += agent.run_loop(False)

        if rewards > best_reward:
            best_alpha = alpha
            best_reward = rewards

    return best_reward, best_alpha, gamma


def CreatePolicyTable(policy_Qtable):
    policy_table = np.array(list(policy_Qtable.items()), dtype=dict)
    for i, _ in enumerate(policy_table):
        non_zero_state_dict = {k: v for k, v in policy_table[i][1].items() if v != 0}
        policy_table[i][1] = max(non_zero_state_dict, key=non_zero_state_dict.get)

    retval = np.empty((22, 12), dtype=str)
    for state in policy_table:
        retval[state[0][0]][state[0][1]] = state[1]

    return retval


def autoValidation():
    gammas = np.arange(0.1, 1, 0.05)
    mp_pool = mp.Pool(os.cpu_count())
    results = mp_pool.imap_unordered(validation, gammas)
    best_result = max(results)
    return best_result


def objective(trial):
    n_learning = 30 * MILLION
    gamma = 1
    alpha = trial.suggest_float("alpha", low=1e-6, high=1e-1, log=True)
    epsilon = trial.suggest_float("epsilon", low=0.5, high=1, step=0.1)
    f_epsilon = trial.suggest_float("f_epsilon", low=0, high=0.3)
    decay_rate = 1 - trial.suggest_float("decay_rate", low=0.1e-4, high=1e-10, log=True)

    agent = QAgent(alpha, gamma, epsilon, f_epsilon, decay_rate)
    rewards = 0
    for i in range(0, (2 * n_learning) // 3):
        agent.update_epsilon()
        agent.run_loop(True)
    agent.Game.shoe.rebuild()
    for _ in range(0, n_learning // 3):
        reward, _ = agent.run_loop(False)
        rewards += reward

    return rewards


def learnOptuna():
    sqlite = create_connection('BJ_30M.db')

    study = optuna.create_study(direction="maximize", storage='sqlite:///BJ_30M.db', load_if_exists=True,
                                study_name='optuna')
    with parallel_backend('multiprocessing'):  # Overrides `prefer="threads"` to use multiprocessing.
        study.optimize(objective, n_trials=500, n_jobs=1, show_progress_bar=True)

    result = study.best_value
    gamma = study.best_params['gamma']
    best_epsilon = study.best_params['epsilon']
    best_alpha = study.best_params['alpha']
    best_f_epsilon = study.best_params['f_epsilon']
    best_decay_rate = study.best_params['decay_rate']
    return result, best_alpha, best_epsilon, best_f_epsilon, best_decay_rate


def create_connection(path):
    connection = sqlite3.connect(path)
    print("Connection to SQLite DB successful")

    return connection

def plotResults(agent, obj_list, hand_num_list):
    obj_list = np.array(obj_list)
    hand_num_list = np.array(hand_num_list)
    obj_list /= -hand_num_list

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(hand_num_list, obj_list)
    if agent.basicStrategy:
        ax.set_title('House Edge\nBasic Strategy')
    else:
        ax.set_title('House Edge\n'+r'$\gamma={' + f'{1:g}' + r'}$, $\alpha={' + f'{agent.alpha:g}' + r'}$')
    ax.set_xlabel('Hand')
    ax.set_ylabel('Reward/Hands')
    ax.set_ylim(0, 0.007)
    ax.grid()
    fig.savefig("plot_bs_" + str(agent.basicStrategy) + "2" +".png", bbox_inches="tight")
    #fig.savefig("plot_bs_" + str(agent.basicStrategy) + "1" +".png")
    plt.show()

def color_table(val):
    if val =='H':
        color = 'red'
    elif val == 'S':
        color = 'yellow'
    elif val == 'D':
        color = 'blue'
    elif val == 'X':
        color = 'white'
    else: #'P'
        color = 'green'

    return 'color: %s' % color
    
def finalTest(best_alpha, best_gamma, best_epsilon, best_final_epsilon, best_decay_rate, basicStrategy=False, learnLateSplit=False):
    n_train = 1 * MILLION
    n_test = 10 * MILLION
    onlyPairs = learnLateSplit

    agent = QAgent(best_alpha, best_gamma, best_epsilon, best_final_epsilon, best_decay_rate, basicStrategy)
    if not basicStrategy:
        agent.train(n_train)
    if learnLateSplit:
        agent.Q_table[2] = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])
        # agent.gamma = 0.1
        # agent.alpha = 0.001
        agent.epsilon = 0.5
        # agent.final_epsilon = 0.2
        agent.train(n_train//5, onlyPairs)

    wins_rate, rewards, obj_list, hand_num_list= agent.test(n_test)

    print(agent.countDict)

    cell_hover = {
    "selector": "td:hover",
    "props": [("background-color", "#FFFFE0")]
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;"
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #a3a2a2; color: white;"
    }

    properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

    print('\t\t\tHard')
    hard_policy = CreatePolicyTable(agent.Q_table[0])
    print(pd.DataFrame(hard_policy[4:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(4, 22))))

    hard = pd.DataFrame(hard_policy[4:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(4, 22))).style.applymap(color_table)
    html = hard.format(precision=2).set_table_styles([cell_hover, index_names, headers]).set_properties(**properties).to_html()
    
    # write html to file
    text_file = open("hard.html", "w")
    text_file.write(html)
    text_file.close()

    print()

    print('\t\t\tSoft')
    soft_policy = CreatePolicyTable(agent.Q_table[1])
    print(pd.DataFrame(soft_policy[13:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(13, 22))))

    soft = pd.DataFrame(soft_policy[13:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(13, 22))).style.applymap(color_table)
    html = soft.format(precision=2).set_table_styles([cell_hover, index_names, headers]).set_properties(**properties).to_html()
    
    # write html to file
    text_file = open("soft.html", "w")
    text_file.write(html)
    text_file.close()

    print()

    print('\t\t\tSplit')
    split_policy = CreatePolicyTable(agent.Q_table[2])
    print(pd.DataFrame(split_policy[2:12, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(2, 12))))
    # , rows=['2,2', '3,3', '4,4', '5,5', '6,6', '7,7', '8,8', '9,9', '10,10', 'A,A']

    split = pd.DataFrame(split_policy[2:12, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(2, 12))).style.applymap(color_table)
    html = split.format(precision=2).set_table_styles([cell_hover, index_names, headers]).set_properties(**properties).to_html()
    
    # write html to file
    text_file = open("split.html", "w")
    text_file.write(html)
    text_file.close()

    print(f"basicStrategy = {basicStrategy}\tlearnLateSplit = {learnLateSplit}\t onlyPairs = {onlyPairs}")
    print(f"Win rate: {wins_rate}")
    print(f"Rewards : {rewards}")

    plotResults(agent, obj_list, hand_num_list)


def color_table(x):
    if (x == 'P'):
        return "background-color: #2eee34; color: black; font-weight: bold;" 
    elif (x == 'S'):
        return "background-color: #eefa07; color: black; font-weight: bold;" 
    elif (x == 'H'):
        return "background-color: #fa0000; color: black; font-weight: bold;" 
    elif (x == 'D'):
        return "background-color: #0000fa; color: black; font-weight: bold;" 
    elif (x == 'X'):
        return "background-color: #ffffff; color: black; font-weight: bold;"

def main():
    """ Rewards:
            Regular WIN     = 1
            DoubleDown WIN  = 2
            SPLIT WIN       = 1 (per win)
            Lose            = -1
            SPLIT Lose      = 1 (per win)
            Surrender Lose  = -0.5
        States:
            (0 = Hard / 1 = Soft / 2 = Splittable , HandSum, DealerSum)
    """

    # best_result = learnOptuna()
    # best_result = [-56267.0, 0.0004999203318563665, 1, 0.29598784420584845, 1- (1.2343289349378915e-09)]
    # best_result = [-56267.0, 0.00030028805421964045, 1, 0.2826914780047174, 1- (6.052041223193185e-10)]
    # best_result = [-56267.0, 0.00030028805421964045, 1, 1, 1]
    # best_result = [-53761.0, 0.0006583044027324456, 1, 26654425630519774, 1 - (4.881918202338584e-10)]
    # best_result = [-52376.0, 0.0005720263618453814, 1, 0.14986966254235395, 1 - (5.796720039295601e-10)]
    best_result = [-50775, 0.0002903031868228513, 1, 0.1706443333378505, 1 - (1.4356609226904362e-10)]
    # best_result = [-43221.0, 0.0009673773214241886, 1, 0.2, 1 - (5.796720039295601e-10)]
    # best_result = [-40615.5, 0.002558204095711428, 1, 0.2, 1 - (5.796720039295601e-10)]
    # best_result = [-36578.0, 0.0010290963909385185, 1, 0.2, 1 - (5.796720039295601e-10)]
    # best_result = [-34776.0, 0.0021999140060916115, 1, 0.2, 1 - (5.796720039295601e-10)]

    print(best_result)
    best_gamma = 1
    best_alpha = best_result[1]
    best_epsilon = best_result[2]
    best_final_epsilon = best_result[3]
    best_decay_rate = best_result[4]

    finalTest(best_alpha, best_gamma, best_epsilon, best_final_epsilon, best_decay_rate, basicStrategy=True, learnLateSplit=False)
    
if __name__ == '__main__':
    main()
