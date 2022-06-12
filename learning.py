import simulator as sym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

action_dict = {3 :'H', 1 : 'S', 2 : 'P', 0 : 'D', 4 : 'X'}

class QAgent:

    def __init__(self, alpha, gamma):
        #The BJ Simulator
        self.Game = sym.Game()

        #Init the table with q_values: HARD(18X10) X SOFT(9X10) X SPLIT(8X10) X ACTION(5)
        # self.Q_table = np.zeros((2,21,13,5))
        self.Q_table = np.zeros((2,33,14,5)) #33 for all sums that are reachable, 14, for 1-13
        self.alpha = alpha

        self.gamma = gamma

    def get_action(self, game_state, player_state, dealer_state):
        return np.argmax(self.Q_table[int(game_state), player_state, dealer_state])
    

    def update_paramaters(self, game_state, player_state , dealer_state , action, reward, next_game_state,next_player_state,next_dealer_state):
        
        #Q-Learn formula
        old_value = self.Q_table[int(game_state), player_state, dealer_state, action]
        next_max = np.max(self.Q_table[int(next_game_state) ,next_player_state, next_dealer_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        #Update the Q_table
        self.Q_table[int(game_state), player_state, dealer_state, action] = new_value

def main():
    ''' Rewards:
            Regular WIN     = 1
            DoubleDown WIN  = 2
            SPLIT WIN       = 1 (per win)
            Lose            = -1
            SPLIT Lose      = 1 (per win)
            Surrender Lose  = -0.5
    '''

    '''
        States:
            (0 = Hard / 1 = Soft / 2 = Splitable , HandSum, DealerSum)
    '''
    n_episodes = 1000000
    alpha = 0.01
    gamma = 0.3
    epsilon = 0.5

    agent = QAgent(alpha, gamma)
    
    wins = 0

    for i in tqdm(range(0,n_episodes)):
        #Reset the game to random state
        game_state, player_state, dealer_state = agent.Game.reset_hands()
        reward = 0
        done = False
        epsilon *= 0.99999
        while (not done):
            # action = {3 :'H', 1 : 'S', 2 : 'P', 0 : 'D', 4 : 'X'}
            if(random.uniform(0, 1) < epsilon):
                #Random Action
                action = int(random.randint(0, 4))
            else:
                action = agent.get_action(game_state,player_state,dealer_state)

            next_game_state, next_player_state, next_dealer_state, reward, done = agent.Game.step(action_dict[action])

            agent.update_paramaters(game_state, player_state, dealer_state ,action,reward,next_game_state,next_player_state,next_dealer_state)

            game_state = next_game_state
            player_state = next_player_state
            dealer_state = next_dealer_state
        
        if(reward > 0):
            wins+= reward
        print(wins/(i+1))

    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')
    # plt.show()
    # table = np.max(agent.Q_table[0,:,:])
    print(pd.DataFrame(np.argmax(agent.Q_table[0],axis=2)))

if __name__ == '__main__':
	main()

