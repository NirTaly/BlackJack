import math
import os
import itertools
import simulator as sym
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import optuna
# from joblib import parallel_backend
import sqlite3

action_dict = {0: 'S', 1: 'H', 2: 'X', 3: 'D', 4: 'P'}

# function that handle case of early BJ
def handleBJ(agent,explore):
    reward, done  = 0, False
    player_state, game_state = agent.Game.sum_hands(agent.Game.playerCards[0])
    dealer_sum, _ = agent.Game.sum_hands(agent.Game.dealerCards)
    if player_state == 21 or dealer_sum == 21:
        action = 'S'
        reward = agent.Game.rewardHandler(dealer_sum, [player_state])
        reward = 1.5 * reward if (reward > 0) else reward
        done = True
        if explore:
            agent.update_parameters(game_state, player_state, agent.Game.dealer_state, action, reward, game_state,
                                player_state)

    return reward, done

def run_loop(agent, explore):
    game_state, player_state = agent.Game.reset_hands()
    reward,done = handleBJ(agent, explore)
    agent.update_epsilon()
    last_split_hands_parameters = []
    for i, _ in enumerate(agent.Game.playerCards):
        while not done:
            action = agent.get_action(game_state, player_state, agent.Game.dealer_state, explore)

            next_game_state, next_player_state, reward, done = agent.Game.step(action)

            if explore:
                if done and agent.Game.isSplit:
                    last_split_hands_parameters.append(
                        [game_state, player_state, action, next_game_state, next_player_state])
                else:
                    agent.update_parameters(game_state, player_state, agent.Game.dealer_state, action, reward,
                                            next_game_state, next_player_state)

                if action == "P":
                    if agent.Game.playerCards[0][0] == 1 and agent.Game.splitAcesAndDone:
                        next_player_state, next_game_state = agent.Game.sum_hands(agent.Game.playerCards[1])
                        last_split_hands_parameters.append(
                            [game_state, player_state, action, next_game_state, next_player_state])
                        break
                    else:
                        sec_hand_player_state, sec_hand_game_state = agent.Game.sum_hands(agent.Game.playerCards[1])
                        agent.update_parameters(game_state, player_state, agent.Game.dealer_state, action, reward,
                                                sec_hand_game_state, sec_hand_player_state)

            game_state = next_game_state
            player_state = next_player_state

        if agent.Game.isSplit:
            if agent.Game.playerCards[0][0] == 1 and agent.Game.splitAcesAndDone:
                break
            player_state, game_state = agent.Game.sum_hands(agent.Game.playerCards[agent.Game.currHand])
            done = False
        else:
            rewards = reward

    if agent.Game.isSplit:
        if explore:
            for i in range(2):
                game_state, player_state, action, next_game_state, next_player_state = last_split_hands_parameters[i]
                agent.update_parameters(game_state, player_state, agent.Game.dealer_state, action, reward[i],
                                        next_game_state, next_player_state)
        rewards = sum(reward)
    return rewards


def CreateQTable(player_state_tuple, legal_actions):
    Q_table = dict()
    start, stop, step = player_state_tuple
    tuple_player_dealer_list = list(itertools.product(list(range(start, stop, step)), list(range(2, 12))))
    for key in tuple_player_dealer_list:
        Q_table[key] = dict.fromkeys(legal_actions, 0)

    return Q_table

def initBasicStategy(rules):
    if rules == 1:
        return initBasicStategy1()
    else:
        pass
def initBasicStategy1():
    Q_table_hard = CreateQTable((4, 22, 1), ['S', 'H', 'X', 'D'])
    Q_table_soft = CreateQTable((13, 22, 1), ['S', 'H', 'X', 'D'])
    Q_table_split = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])

    # -- HARD --

    # 4-8 against 2-10 *HIT*
    for p in range(4,12):
        for d in range(2,12):
            Q_table_hard[(p,d)]['H'] = 1

    # 9 against 3-6 *DOUBLE*
    for d in range(3,7):
        Q_table_hard[(9,d)]['D'] = 2

    # 10,11 against 2-9 *DOUBLE*
    for p in range(10,12):
        for d in range(2,10):
            Q_table_hard[(p,d)]['D'] = 2

    # 11 against 10 *DOUBLE*
    Q_table_hard[(11, 10)]['D'] = 2

    # 10,11 against 10,A *HIT*
    Q_table_hard[(10, 10)]['H'] = 1
    Q_table_hard[(10, 11)]['H'] = 1
    Q_table_hard[(11, 11)]['H'] = 1

    # 12 against 2,3,7,8,9,10 *HIT*
    for d in [2,3,7,8,9,10,11]:
        Q_table_hard[(12,d)]['H'] = 1

    # 12 against 4-6 *STAND*
    for d in range(4,7):
        Q_table_hard[(12,d)]['S'] = 1

    # 13-16 against 2-6 *STAND*
    for p in range(13,17):
        for d in range(2,7):
            Q_table_hard[(p,d)]['S'] = 1

    # 13 against 7-10 *HIT*
    for p in range(12, 17):
        for d in range(7,12):
            Q_table_hard[(p,d)]['H'] = 1

    # 14,15 against 10,A *SURRENDER*
    for d in range(9,12):
        Q_table_hard[(16,d)]['X'] = 2
    Q_table_hard[(15, 10)]['X'] = 2

    # 17 against 2-10 *STAND*
    for d in range(2,12):
        Q_table_hard[(17,d)]['S'] = 1

    # 18+ against ALL *STAND*
    for p in range(18,22):
        for d in range(2,12):
            Q_table_hard[(p,d)]['S'] = 1


    # hard_policy = CreatePolicyTable(Q_table_hard)
    # print(pd.DataFrame(hard_policy[4:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(4, 22))))

    # -- SOFT --
    # 13,14 against 2-4,7-11 *HIT*
    for p in range(13,18):
        for d in range(2,12):
            Q_table_soft[(p,d)]['H'] = 1
    
    # 13,14 against 5,6 *DOUBLE*
    for p in range(13,15):
        for d in range(5,7):
            Q_table_soft[(p,d)]['D'] = 2

    # 15,16 against 4,5,6 *DOUBLE*
    for p in range(15,17):
        for d in range(4,7):
            Q_table_soft[(p,d)]['D'] = 2

    # 17,18 against 3-6 *DOUBLE*
    for p in range(17,19):
        for d in range(3,7):
            Q_table_soft[(p,d)]['D'] = 2

    # 18 against 2,7,8 *STAND*
    for d in range(2,9):
            Q_table_soft[(18,d)]['S'] = 1

    # 18 against 9-11 *HIT*
    for d in range(9,12):
            Q_table_soft[(18,d)]['H'] = 1
            
    # 19-21 against ALL *STAND*
    for p in range(19,22):
        for d in range(2,12):
            Q_table_soft[(p,d)]['S'] = 1

    # -- SPLIT --
    
    # 2,3 against 2,3,8,9,10,A *HIT*
    for p in range(2,4):
        for d in [2,3,8,9,10,11]:
            Q_table_split[(p,d)]['H'] = 1

    # 2,3 against 4-7 *SPLIT*
    for p in range(2,4):
        for d in range(4,8):
            Q_table_split[(p,d)]['P'] = 1

    # correct 3 against A *SURRENDER*
    Q_table_split[(3,11)]['X'] = 1

    # 4 against ALL *HIT*
    for d in range(2,12):
        Q_table_split[(4,d)]['H'] = 1

    # 5 against 2-9 *DOUBLE*
    for d in range(2,10):
        Q_table_split[(5,d)]['D'] = 2

    
    # 5 against 10,11 *HIT*
    for d in range(2,12):
        Q_table_split[(5,d)]['H'] = 1

    # 6 against 2,7,8,9,10 *HIT*
    for d in [2,7,8,9,10,11]:
        Q_table_split[(6,d)]['H'] = 1

    # 6 against 3,4,5,6 *SPLIT*
    for d in [3,4,5,6]:
        Q_table_split[(6,d)]['P'] = 1

    # 7 against 8-A *HIT*
    for d in [8,9,10,11]:
        Q_table_split[(7,d)]['H'] = 1

    # 7 against 2,3,4,5,6,7 *SPLIT*
    for d in [2,3,4,5,6,7]:
        Q_table_split[(7,d)]['P'] = 1
    
    # 8 against 2,3,4,5,6,7,8,9 *SPLIT*
    for d in [2,3,4,5,6,7,8,9,10,11]:
        Q_table_split[(8,d)]['P'] = 1

    # 9 against 2,3,4,5,6,8,9 *SPLIT*
    for d in [2,3,4,5,6,8,9]:
        Q_table_split[(9,d)]['P'] = 1
    
    # 9 against A *STAND*
    for d in [7,10,11]:
        Q_table_split[(9,d)]['S'] = 1

    # 10 against ALL *STAND*
    for d in range(2,12):
        Q_table_split[(10,d)]['S'] = 1

    # A against 2-10 *SPLIT*
    for d in range(2,12):
        Q_table_split[(11,d)]['P'] = 1

    return [Q_table_hard,Q_table_soft,Q_table_split]


class QAgent:

    def __init__(self, alpha, gamma, epsilon, basicStrategy=False):
        # The BJ Simulator
        self.Game = sym.Game()

        if not basicStrategy:
            # Init the table with q_values: HARD(18X10) X SOFT(9X10) X SPLIT(8X10) X ACTION(5)
            # 'SD' = stand after double - pseudo state so don't mix stand-after-hit with stand-after-double
            Q_table_hard = CreateQTable((4, 22, 1), ['S', 'H', 'X', 'D'])
            Q_table_soft = CreateQTable((13, 22, 1), ['S', 'H', 'X', 'D'])
            Q_table_split = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])
            Q_table_SD = CreateQTable((6, 22, 1), ['SD'])
            self.Q_table = [Q_table_hard, Q_table_soft, Q_table_split, Q_table_SD]
        else:
            self.Q_table = initBasicStategy(1)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _playerStateFromGameState(self, game_state, player_state):
        if game_state == 2:
            if self.Game.playerCards[0][0] == 1:
                player_state = 11
            else:
                player_state //= 2
        return player_state

    def update_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon *= 0.999999
        else:
            self.epsilon = 0.05

    def get_action(self, game_state, player_state, dealer_state, explore=True):
        if (random.uniform(0, 1) < self.epsilon and explore):
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
        next_player_state = self._playerStateFromGameState(next_game_state, next_player_state)

        # Q-Learn formula
        old_value = self.Q_table[game_state][(player_state, dealer_state)][action]
        next_max = 0.0
        double_reward = reward
        if (next_player_state > 21 or action in {'S', 'X'}):  # terminal state
            next_max = 0.0
        elif action == 'D': #next valid states is only 'S'
            reward = 0
            next_max = self.Q_table[3][(next_player_state, dealer_state)]['SD']
        else:
            next_max = max(
                itertools.islice(self.Q_table[next_game_state][(next_player_state, dealer_state)].values(), 2))
            # next_max = max(self.Q_table[next_game_state][(next_player_state, dealer_state)].values())
        new_value = self.alpha * (reward + self.gamma * next_max - old_value)

        # Update the Q_table
        self.Q_table[game_state][(player_state, dealer_state)][action] += new_value
        if next_player_state <= 21 and action == 'D':
            old_value_double = next_max
            new_value_double = self.alpha * (double_reward - old_value_double)
            self.Q_table[3][(next_player_state, dealer_state)]['SD'] += new_value_double

    def train(self, n_train):
        for _ in tqdm(range(0, n_train)):
            run_loop(self, True)

    def test(self, n_test):
        wins = 0
        hands = 0
        rewards = 0
        self.Game.shoe.rebuild()
        for _ in tqdm(range(0, n_test)):
            reward = run_loop(self, False)
            rewards += reward
            wins += (reward > 0) + (reward == 2 and self.Game.isSplit)
            hands += 1 + self.Game.isSplit
        return wins / hands, rewards


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
        for _ in tqdm(range(0, (9 * n_learning) // 10)):
            run_loop(agent, True)
        for _ in tqdm(range(0, n_learning // 10)):
            rewards += run_loop(agent, False)

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
    n_learning = 25000000
    alpha = trial.suggest_float("alpha", low=1e-6, high=1e-4, log=True)
    gamma = trial.suggest_float("gamma", low=0.2, high=1)
    # epsilon = trial.suggest_float("epsilon",low=0.7,high=1,step=0.1)
    epsilon = 0.9

    agent = QAgent(alpha, gamma, epsilon)
    rewards = 0
    for _ in range(0, (9 * n_learning) // 10):
        run_loop(agent, True)
    agent.Game.shoe.rebuild()
    for _ in range(0, n_learning // 10):
        rewards += run_loop(agent, False)

    return rewards


def learnOptuna():
    sqlite = create_connection('Optuna6.db')

    study = optuna.create_study(direction="maximize", storage='sqlite:///Optuna6.db', load_if_exists=True,
                                study_name='distributed-example')
    with parallel_backend('multiprocessing'):  # Overrides `prefer="threads"` to use multiprocessing.
        study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)

    result = study.best_value
    gamma = study.best_params['gamma']
    epsilon = study.best_params['epsilon']
    best_alpha = study.best_params['alpha']
    return result, best_alpha, gamma, epsilon


def create_connection(path):
    connection = sqlite3.connect(path)
    print("Connection to SQLite DB successful")

    return connection


def finalTest(best_alpha, best_gamma, best_epsilon, basicStrategy=False):
    n_train = 10000000
    n_test = 100000000

    agent = QAgent(best_alpha, best_gamma, best_epsilon, basicStrategy)
    if not basicStrategy:
        agent.train(n_train)
    wins_rate, rewards = agent.test(n_test)

    print('\t\t\tHard')
    hard_policy = CreatePolicyTable(agent.Q_table[0])
    print(pd.DataFrame(hard_policy[4:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(4, 22))))

    print()

    print('\t\t\tSoft')
    soft_policy = CreatePolicyTable(agent.Q_table[1])
    print(pd.DataFrame(soft_policy[13:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(13, 22))))

    print()

    print('\t\t\tSplit')
    split_policy = CreatePolicyTable(agent.Q_table[2])
    print(pd.DataFrame(split_policy[2:12, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(2, 12))))
    # , rows=['2,2', '3,3', '4,4', '5,5', '6,6', '7,7', '8,8', '9,9', '10,10', 'A,A']

    print(f"Win rate: {wins_rate}")
    print(f"Rewards : {rewards}")


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
    # best_result = autoValidation()
    # best_result = [[-148636.5, 0.5, 0.0001, 0.9], [-139052.5, 0.5, 4.8e-05, 1], [-135307.0, 0.5, 5.5e-05, 0.7], [-131909.5, 0.5, 4.1e-5, 0.7],
    # [-129282.0, 0.5, 5.2e-05, 1], [-129224.0, 0.5, 6.5e-5, 0.9], [-123478.0, 0.5, 4.2e-05, 1], [-86508.0, 0.3, 7.7e-05, 0.9], [-71173.5, 0.9616570203641142, 0.00196250102232301, 0.8073720122385102]]
    # -73996.0 and parameters: {'alpha': 0.001492950854301212, 'gamma': 0.9131572889821157,'epsilon': 0.3741137556539688}.
    # -67898.5 and parameters: {'alpha': 0.0004803668759456502, 'gamma': 1.1962376455954642, 'epsilon': 0.6664046431453283}.
    # {'alpha': 0.001098144772126204, 'gamma': 1.074120810719544, 'epsilon': 0.9072751496865936}
    # -66680.5 and parameters: {'alpha': 0.0006297423928461678, 'gamma': 1.1792443587862564, 'epsilon': 0.8986248216065134}
    best_result = [-65368.0, 1.1792443587862564, 0.000942578570381582, 0.9210472074709081]

    print(best_result)
    best_gamma = best_result[1]
    best_alpha = best_result[2]
    best_epsilon = best_result[3]

    finalTest(best_alpha, best_gamma, best_epsilon, basicStrategy=False)



if __name__ == '__main__':
    main()
