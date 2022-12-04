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
#from joblib import parallel_backend
import sqlite3

action_dict = {0: 'S', 1: 'H', 2: 'X', 3: 'D', 4: 'P'}

def run_loop(agent, explore):
    game_state, player_state = agent.Game.reset_hands()
    done = False
    rewards = 0
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
    # add dummy for burned states
    Q_table["BURNED"] = 0
    return Q_table


class QAgent:

    def __init__(self, alpha, gamma, epsilon):
        # The BJ Simulator
        self.Game = sym.Game()

        # Init the table with q_values: HARD(18X10) X SOFT(9X10) X SPLIT(8X10) X ACTION(5)

        Q_table_hard = CreateQTable((4, 22, 1), ['S', 'H', 'X', 'D'])
        Q_table_soft = CreateQTable((13, 22, 1), ['S', 'H', 'X', 'D'])
        Q_table_split = CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])

        self.Q_table = [Q_table_hard, Q_table_soft, Q_table_split]
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

    def update_epsilon(self):  # TODO add to optuna
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
                # return np.argmax(self.Q_table[game_state, player_state, dealer_state][0:3])
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
        if (next_player_state > 21 or action in {'S','X'}): #terminal state
            next_max = self.Q_table[next_game_state]["BURNED"]
        elif action == 'D': #next valid states is only 'S'
            next_max = self.Q_table[next_game_state][(next_player_state, dealer_state)]['S']        #TODO validate that shouldnt be 0
        else:
            next_max = max(itertools.islice(self.Q_table[next_game_state][(next_player_state, dealer_state)].values(), 2))
            # next_max = max(self.Q_table[next_game_state][(next_player_state, dealer_state)].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        # Update the Q_table
        self.Q_table[game_state][(player_state, dealer_state)][action] = new_value

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
        return wins/hands, rewards


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
    policy_table = np.delete(policy_table, len(policy_table) - 1, axis=0)
    for i, _ in enumerate(policy_table):
        policy_table[i][1] = max(policy_table[i][1], key=policy_table[i][1].get)

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
    alpha = trial.suggest_float("alpha", low=1e-6, high=1e-4, step=1e-6)
    gamma = trial.suggest_float("gamma", low=0.2, high=0.9, step=0.05)
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
    with parallel_backend('multiprocessing'):  # Overrides `prefer="threads"` to use multi-processing.
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


def finalTest(best_alpha, best_gamma, best_epsilon):
    n_train = 25000000
    n_test = 2500000

    agent = QAgent(best_alpha, best_gamma, best_epsilon)
    agent.train(n_train)
    wins_rate, rewards = agent.test(n_test)

    print('\t\t\tHard')
    hard_policy = CreatePolicyTable(agent.Q_table[0])
    print(pd.DataFrame(hard_policy[5:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(5, 22))))

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

    #best_result = learnOptuna()
    # best_result = autoValidation()
    # best_result = [[-148636.5, 0.5, 0.0001, 0.9], [-139052.5, 0.5, 4.8e-05, 1], [-135307.0, 0.5, 5.5e-05, 0.7], [-131909.5, 0.5, 4.1e-5, 0.7],
    # [-129282.0, 0.5, 5.2e-05, 1], [-129224.0, 0.5, 6.5e-5, 0.9], [-123478.0, 0.5, 4.2e-05, 1], [-86508.0, 0.3, 7.7e-05, 0.9], [-71173.5, 0.9616570203641142, 0.00196250102232301, 0.8073720122385102]]
    #-73996.0 and parameters: {'alpha': 0.001492950854301212, 'gamma': 0.9131572889821157,'epsilon': 0.3741137556539688}.
    best_result = [-73996.0, 0.9131572889821157, 0.001492950854301212, 0.3741137556539688]

    print(best_result)
    best_gamma = best_result[1]
    best_alpha = best_result[2]
    best_epsilon = best_result[3]

    finalTest(best_alpha, best_gamma, best_epsilon)


if __name__ == '__main__':
    main()
