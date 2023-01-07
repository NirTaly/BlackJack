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
        for i in range(2):
            game_state, player_state, action, next_game_state, next_player_state = hands_params[i]
            next_max = max(itertools.islice(
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

        return rewards, wins

    def train(self, n_train, onlyPairs=False):
        for i in tqdm(range(0, n_train)):
            self.update_epsilon()
            self.run_loop(True,onlyPairs)

    def test(self, n_test):
        total_wins = 0
        hands = 0
        rewards = 0
        self.Game.shoe.rebuild()
        for _ in tqdm(range(0, n_test)):
            reward, wins = self.run_loop(False)
            rewards += reward
            total_wins += wins
            hands += 1 + self.Game.isSplit
        return total_wins / hands, rewards


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
                                study_name='InalDinak')
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


def finalTest(best_alpha, best_gamma, best_epsilon, best_final_epsilon, best_decay_rate, basicStrategy=False, learnLateSplit=False):
    n_train = 20 * MILLION
    n_test = 10 * MILLION

    agent = QAgent(best_alpha, best_gamma, best_epsilon, best_final_epsilon, best_decay_rate, basicStrategy)
    if not basicStrategy:
        agent.train(n_train)
    if learnLateSplit :
        agent.Q_table[2] = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])
        agent.gamma = 0.1
        agent.alpha = 0.001
        agent.epsilon = 0.7
        agent.final_epsilon = 0.2
        agent.train(n_train//3, onlyPairs=True)

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
    # -73996.0 and parameters: {'alpha': 0.001492950854301212, 'gamma': 0.9131572889821157,'epsilon': 0.3741137556539688}.
    # -67898.5 and parameters: {'alpha': 0.0004803668759456502, 'gamma': 1.1962376455954642, 'epsilon': 0.6664046431453283}.
    # {'alpha': 0.001098144772126204, 'gamma': 1.074120810719544, 'epsilon': 0.9072751496865936}
    # -66680.5 and parameters: {'alpha': 0.0006297423928461678, 'gamma': 1.1792443587862564, 'epsilon': 0.8986248216065134}
    # -10076.0 and parameters: {'alpha': 0.0023883300020830127, 'gamma': 0.7982205072789618, 'epsilon': 0.8}.
    # -10071.0 and parameters: {'alpha': 0.0020742381406755835, 'gamma': 0.445944098197451, 'epsilon': 0.8}.
    # -9686.0 and parameters: {'alpha': 0.002721225490977786, 'gamma': 0.7737094148769824, 'epsilon': 0.7}.
    # -107871.5 and parameters: {'alpha': 0.004993279721394013, 'gamma': 0.7798829655701662, 'epsilon': 1.0}.
    # -97437.0 and parameters: {'alpha': 0.0014432395677060068, 'gamma': 0.7491076134854117, 'epsilon': 0.9000000000000001}
    # -84888.0 and parameters: {'alpha': 0.0020991386306812507, 'gamma': 0.9306406087558762, 'epsilon': 1.0}.
    # -83525.5 and parameters: {'alpha': 0.0013426092343642878, 'gamma': 0.8720959544864512, 'epsilon': 1.0}.
    # value: -61062.5 and parameters: {'alpha': 0.0016639830031509864, 'epsilon': 1.0}
    # -59056.0 and parameters: {'alpha': 0.0004090330318976022, 'epsilon': 1.0, 'f_epsilon': 0.021475900072008624, 'decay_rate': 1.4757112667433222e-09}
    # -56423.5 and parameters: {'alpha': 0.0004999203318563665, 'epsilon': 1.0, 'f_epsilon': 0.29598784420584845, 'decay_rate': 1.2343289349378915e-09}
    # best_result = [-56267.0, 0.0004999203318563665, 1, 0.29598784420584845, 1- (1.2343289349378915e-09)]
    best_result = [-56267.0, 0.00030028805421964045, 1, 0.2826914780047174, 1- (6.052041223193185e-10)]
    # best_result = [-56267.0, 0.00030028805421964045, 1, 1, 1]

    print(best_result)
    best_gamma = 1
    best_alpha = best_result[1]
    best_epsilon = best_result[2]
    best_final_epsilon = best_result[3]
    best_decay_rate = best_result[4]

    finalTest(best_alpha, best_gamma, best_epsilon, best_final_epsilon, best_decay_rate, basicStrategy=False, learnLateSplit=True)


if __name__ == '__main__':
    main()
