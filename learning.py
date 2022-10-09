import math
import os
import itertools
import simulator as sym
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

action_dict = {0 :'S', 1 : 'H', 2 : 'X', 3 : 'D', 4 : 'P'}

def run_loop(agent):
    game_state, player_state = agent.Game.reset_hands()
    done = False
    agent.update_epsilon()
    while not done:
        action = agent.get_action(game_state,player_state,agent.Game.dealer_state)

        next_game_state, next_player_state, reward, done = agent.Game.step(action)

        agent.update_parameters(game_state, player_state, agent.Game.dealer_state, action, reward, next_game_state,
                                next_player_state)

        game_state = next_game_state
        player_state = next_player_state
    return reward

def CreateQTable(player_state_tuple,legal_actions):
    Q_table = dict()
    start, stop, step = player_state_tuple
    tuple_player_dealer_list = list(itertools.product(list(range(start,stop,step)), list(range(2, 12))))
    for key in tuple_player_dealer_list:
        Q_table[key] = dict.fromkeys(legal_actions, 0)
    # add dummy for burned states
    Q_table["BURNED"] = -1
    return Q_table

class QAgent:

    def __init__(self, alpha, gamma, epsilon):
        # The BJ Simulator
        self.Game = sym.Game()

        #Init the table with q_values: HARD(18X10) X SOFT(9X10) X SPLIT(8X10) X ACTION(5)

        Q_table_hard = CreateQTable((4,22,1) , ['S','H','X','D'])
        Q_table_soft = CreateQTable((13,22,1), ['S','H','X','D'])
        Q_table_split= CreateQTable((2,12,1) , ['S','H','X','D','P'])

        self.Q_table = [Q_table_hard, Q_table_soft, Q_table_split]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def update_epsilon(self):
        self.epsilon *= 0.99999

    def get_action(self, game_state, player_state, dealer_state, explore=True):
        if(random.uniform(0, 1) < self.epsilon and explore):
            #Random Action - Exploring
            if self.Game.first_move:
                if game_state == 2:
                    return action_dict[random.randint(0, 4)]
                else:
                    return action_dict[random.randint(0,3)]
            else:
                return action_dict[random.randint(0,1)]
        else:
            if self.Game.first_move:
                if game_state == 2:
                    if self.Game.playerCards[0][0] == 1:
                        player_state = 11
                    else:
                        player_state /= 2

                return max(self.Q_table[game_state][(player_state, dealer_state)], key=self.Q_table[game_state][(player_state, dealer_state)].get)
                # return np.argmax(self.Q_table[game_state, player_state, dealer_state][0:3])
            else:
                return max(dict(itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)),
                    key=dict(itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)).get)

    def update_parameters(self, game_state, player_state, dealer_state, action, reward, next_game_state,
                          next_player_state):
        if game_state == 2:
            if self.Game.playerCards[0][0] == 1:
                player_state = 11
            else:
                player_state //= 2

        # Q-Learn formula
        old_value = self.Q_table[game_state][(player_state,dealer_state)][action]
        next_max = 0
        if (next_player_state > 21):
            next_max = self.Q_table[next_game_state]["BURNED"]
        else:
            next_max = max(self.Q_table[next_game_state][(next_player_state,dealer_state)].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        # Update the Q_table
        self.Q_table[game_state][(player_state, dealer_state)][action] = new_value

    def train(self, n_train):
        for _ in tqdm(range(0, n_train)):
            # Reset the game to random state
            run_loop(self)

    def test(self, n_test):
        rewards = 0
        for _ in tqdm(range(0, n_test)):
            # Reset the game to random state
            game_state, player_state = self.Game.reset_hands()
            reward = 0
            done = False
            while not done:
                # action = {0 :'H', 1 : 'S', 2 : 'D', 3 : 'P', 4 : 'X'}
                action = self.get_action(game_state, player_state, self.Game.dealer_state,explore=False)
                next_game_state, next_player_state, reward, done = self.Game.step(action)
                self.update_parameters(game_state, player_state, self.Game.dealer_state, action, reward, next_game_state,
                                       next_player_state)

                game_state = next_game_state
                player_state = next_player_state

            rewards += reward
        return rewards

def validation(alpha):
    """
    best gamma = 0.001
    best alpha = 0.08
    """
    n_learning = 2000000
    gammas = np.arange(0.001, 1, 0.01)

    best_gamma = gammas[0]
    best_reward = -math.inf
    for gamma in gammas:
        agent = QAgent(alpha, gamma, epsilon=0.6)
        rewards = 0
        for _ in tqdm(range(0, (9*n_learning)//10)):
            run_loop(agent)
        for _ in tqdm(range(0, n_learning//10)):
            reward = run_loop(agent)

            rewards += reward
        if rewards > best_reward:
            best_gamma = gamma
            best_reward = rewards

    return best_reward, best_gamma, alpha

def CreatePolicyTable(policy_Qtable):
    policy_table = np.array(list(policy_Qtable.items()), dtype=dict)
    policy_table = np.delete(policy_table, len(policy_table) - 1, axis=0)
    for i, _ in enumerate(policy_table):
        policy_table[i][1] = max(policy_table[i][1], key=policy_table[i][1].get)

    retval = np.empty((22, 12), dtype=str)
    for state in policy_table:
        retval[state[0][0]][state[0][1]] = state[1]

    return retval

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
    n_train = 2000000
    n_test = 300000

    # alphas = np.arange(0.0001, 2, 0.005)
    # mp_pool = mp.Pool(os.cpu_count())
    # results = mp_pool.imap_unordered(validation, alphas)
    # best_result = max(results)

    # best_rewards, best_gamma, best_alpha = validation(0.3)
    best_result = [1636,0.6,0.005, 0.6]

    print(best_result)
    best_gamma = best_result[1]
    best_alpha = best_result[2]
    best_epsilon = best_result[3]

    # Training policy
    agent = QAgent(best_alpha, best_gamma, best_epsilon)
    agent.train(n_train)
    wins = agent.test(n_test)

    print('\t\t\tHard')
    hard_policy = CreatePolicyTable(agent.Q_table[0])
    print(pd.DataFrame(hard_policy[4:,2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(4, 22))))

    print()

    print('\t\t\tSoft')
    soft_policy = CreatePolicyTable(agent.Q_table[1])
    print(pd.DataFrame(soft_policy[13:, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(13, 22))))

    print()

    print('\t\t\tSplit')
    split_policy = CreatePolicyTable(agent.Q_table[2])
    print(pd.DataFrame(split_policy[2:12, 2:], columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'A'], index=list(range(2, 12))))
    # , rows=['2,2', '3,3', '4,4', '5,5', '6,6', '7,7', '8,8', '9,9', '10,10', 'A,A']

    print(wins / n_test)


if __name__ == '__main__':
    main()
