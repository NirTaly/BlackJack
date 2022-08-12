import os

import simulator as sym
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

action_dict = {0: 'H', 1: 'S', 2: 'P', 3: 'D', 4: 'X'}
epsilon = 0.3

def run_loop(agent):
    game_state, player_state, dealer_state = agent.Game.reset_hands()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            # Random Action
            # action = {0 :'H', 1 : 'S', 2 : 'P', 3 : 'D', 4 : 'X'}
            action = int(random.randint(0, 3))
        else:
            action = agent.get_action(game_state, player_state, dealer_state)

        next_game_state, next_player_state, next_dealer_state, reward, done = agent.Game.step(
            action_dict[action])

        agent.update_parameters(game_state, player_state, dealer_state, action, reward, next_game_state,
                                next_player_state, next_dealer_state)

        game_state = next_game_state
        player_state = next_player_state
        dealer_state = next_dealer_state
    return reward

class QAgent:

    def __init__(self, alpha, gamma):
        # The BJ Simulator
        self.Game = sym.Game()

        #Init the table with q_values: HARD(18X10) X SOFT(9X10) X SPLIT(8X10) X ACTION(5)
        # self.Q_table = np.zeros((2,21,13,5))
        self.Q_table = np.zeros((2,33,14,4)) #33 for all sums that are reachable, 14, for 1-13
        # self.Q_table[:,:,:,0:5] = 3
        self.alpha = alpha
        self.gamma = gamma
        self.epislon = 0.3

    def get_action(self, game_state, player_state, dealer_state):
        return np.argmax(self.Q_table[int(game_state), player_state, dealer_state])

    def update_parameters(self, game_state, player_state, dealer_state, action, reward, next_game_state,
                          next_player_state, next_dealer_state):
        # Q-Learn formula
        old_value = self.Q_table[int(game_state), player_state, dealer_state, action]
        next_max = np.max(self.Q_table[int(next_game_state), next_player_state, next_dealer_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        # Update the Q_table
        self.Q_table[int(game_state), player_state, dealer_state, action] = new_value

    def train(self, n_train):
        for _ in tqdm(range(0, n_train)):
            # Reset the game to random state
            run_loop(self)

    def test(self, n_test):
        wins = 0
        for _ in tqdm(range(0, n_test)):
            # Reset the game to random state
            game_state, player_state, dealer_state = self.Game.reset_hands()
            reward = 0
            done = False
            while not done:
                # action = {0 :'H', 1 : 'S', 2 : 'P', 3 : 'D', 4 : 'X'}
                action = self.get_action(game_state, player_state, dealer_state)
                next_game_state, next_player_state, next_dealer_state, reward, done = self.Game.step(
                    action_dict[action])
                self.update_parameters(game_state, player_state, dealer_state, action, reward, next_game_state,
                                       next_player_state, next_dealer_state)

                game_state = next_game_state
                player_state = next_player_state
                dealer_state = next_dealer_state

            if reward > 0:
                wins += reward
        return wins

def validation(alpha):
    """
    best gamma = 0.001
    best alpha = 0.08
    """
    n_learning = 5000
    gammas = np.arange(0.001, 1, 0.01)

    best_gamma = gammas[0]
    best_wins = 0
    for gamma in gammas:
        agent = QAgent(alpha, gamma)
        wins = 0
        for _ in tqdm(range(0, n_learning)):
            reward = run_loop(agent)

            wins += reward > 0
        if wins > best_wins:
            best_gamma = gamma
            best_wins = wins

    return best_wins, best_gamma, alpha

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
    n_train = 100000
    n_test = 100000
    alphas = np.arange(0.0001, 2, 0.005)

    mp_pool = mp.Pool(os.cpu_count())
    results = mp_pool.imap_unordered(validation, alphas)
    # best_rewards, best_gamma, best_alpha = validation(0.3)

    best_result = max(results)
    print (best_result)
    best_gamma = best_result[1]
    best_alpha = best_result[2]
    # Training policy
    agent = QAgent(best_alpha, best_gamma)
    agent.train(n_train)
    wins = agent.test(n_test)

    hard_policy = np.argmax(agent.Q_table[0], axis=2).astype(str)
    hard_policy[hard_policy == '0'] = 'H'
    hard_policy[hard_policy == '1'] = 'S'
    hard_policy[hard_policy == '2'] = 'P'
    hard_policy[hard_policy == '3'] = 'D'
    hard_policy[hard_policy == '4'] = 'X'
    hard_policy = hard_policy[4:22, 1:]
    print(pd.DataFrame(hard_policy, columns=['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'], index=list(range(4, 22))))

    print()
    soft_policy = np.argmax(agent.Q_table[1], axis=2).astype(str)
    soft_policy[soft_policy == '0'] = 'H'
    soft_policy[soft_policy == '1'] = 'S'
    soft_policy[soft_policy == '2'] = 'P'
    soft_policy[soft_policy == '3'] = 'D'
    soft_policy[soft_policy == '4'] = 'X'
    soft_policy = soft_policy[13:22, 1:]
    print(
        pd.DataFrame(soft_policy, columns=['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'], index=list(range(13, 22))))

    print(wins / n_test)


if __name__ == '__main__':
    main()
