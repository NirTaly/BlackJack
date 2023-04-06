from tqdm import tqdm
import learning
import simulator as sim
import basic_strategy as bs
import common
import itertools
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use( 'tkagg' ) 

def roundCount(count):
    return round(count,1)

def initCountDict(game):
    total_min_count_val = common.COUNT_MIN_VAL_DECK * game.shoe.n
    total_max_count_val = common.COUNT_MAX_VAL_DECK * game.shoe.n
    d = dict()

    for running_count in range(total_min_count_val, total_max_count_val+1):
        for i in range(0,10):
            d[round(float(running_count) + 0.1 * i,1)] = (0,0)
    return d

def normalize(d : dict):
    norm = dict()
    for key in d.keys():
        (rewards, hands) = d[key]
        if(hands != 0):
            if hands < 1000:
                norm[key] = 0
            else:
                norm[key] = rewards / hands
            
    return norm

def getOnlyRewards(d : dict):
    only = dict()
    for key in d.keys():
        (rewards, hands) = d[key]
        if(hands != 0):
            only[key] = rewards
    return only

def getOnlyHands(d : dict):
    only = dict()
    for key in d.keys():
        (_, hands) = d[key]
        if(hands != 0):
            only[key] = hands
    return only

def lps_dict(d: dict):
    lps_dict = dict()
    for key in d.keys():
        if abs(key) < 80:
            rounded = roundCount(key)
            if rounded not in lps_dict:
                lps_dict[rounded] = d[key]
            else:
                (lrewards, lhands) = lps_dict[rounded]
                (rrewards, rhands) = d[key]
                lps_dict[rounded] = (lrewards + rrewards, lhands + rhands)
    return lps_dict
class CountAgent:
    def __init__(self,vec=False):
        self.game = sim.Game()
        self.Q_table = bs.initBasicStrategy()
        self.countDict = initCountDict(self.game)
        self.XVecs = []
        self.YVec = []
        self.vec = vec
    
    def handleBJ(self):
        reward, done = 0, False
        player_state, _ = self.game.sum_hands(self.game.playerCards[0])
        dealer_sum, _ = self.game.sum_hands(self.game.dealerCards)
        if player_state == 21 or dealer_sum == 21:
            reward = self.game.rewardHandler(dealer_sum, [player_state])
            if reward > 0:
                self.game.money += (5/2) * self.game.bet
                reward *= 1.5
            done = True
        return reward, done
    
    def getAction(self, game_state, player_state, dealer_state):
        if self.game.first_move:
            player_state = learning.playerStateFromGameState(self.game, game_state, player_state)
            return max(self.Q_table[game_state][(player_state, dealer_state)],
                       key=self.Q_table[game_state][(player_state, dealer_state)].get)

        else:
            return max(dict(itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)),
                       key=dict(
                           itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)).get)

    # function that place the best bet, probably according to Kelly criterion
    def getBet(self,vec):
        count = roundCount(self.game.get_count())
        if count < -30:
            count = -30
        elif count > 30:
            count = 30
        if vec:
            p = 0.5 + common.winrateDictVec[count]
        else:
            p = 0.5 + common.winrateDict[count]
        q = 1 - p
        
        bet = max (self.game.minBet, int(self.game.money * (p - q)))
        return bet
        return 1

    def runLoop(self):
        count = self.game.get_count(vec=self.vec)
        countVec = np.array(self.game.shoe.countVec)
        game_state, player_state = self.game.reset_hands()
        self.game.place_bet(self.getBet(self.vec))
        # print(f"bet = {self.getBet()}")
        # input()
        reward, done = self.handleBJ()
        for i, _ in enumerate(self.game.playerCards):
            while not done:
                action = self.getAction(game_state, player_state, self.game.dealer_state)

                game_state, player_state, reward, done = self.game.step(action)

            if self.game.isSplit:
                if self.game.playerCards[0][0] == 1 and self.game.splitAcesAndDone:
                    break
                player_state, game_state = self.game.sum_hands(self.game.playerCards[self.game.currHand])
                done = False
            else:
                rewards = reward

        if self.game.isSplit:
            rewards = sum(reward)
            wins = sum(1 for w in reward if w > 0)
        else:
            wins = 1 if (reward > 0) else 0

        (count_rewards, touched) = self.countDict[round(count,1)]
        self.countDict[round(count,1)] = (count_rewards + rewards, touched + 1)
        self.XVecs.append(countVec)
        self.YVec.append(rewards)

        return rewards, wins

def batchGames(vec=False):
    # hands = 0

    fig, ax = plt.subplots()
    plt.xlabel('Hands')
    plt.ylabel('Money')
    plt.title("Money - Hands")
    ax.set_yscale('log')
    data = []
    for i in range(5):
        countAgent = CountAgent(vec=vec)
        data.append(dict())
        hands = 0
        data[i][hands] = countAgent.game.money
        while countAgent.game.money > countAgent.game.minBet and hands < 30000:
            print("Money: " + str(countAgent.game.money) + f" , Hands = {hands}")#, end='\r') 
            countAgent.runLoop()
            hands +=1
            if hands % 1000 == 0:
                data[i][hands] = countAgent.game.money
        ax.plot(*zip(*data[i].items()),label=f'{i}')
    
    plt.show()
    


def finalTest(vec = False):
    countAgent = CountAgent(vec)
    for _ in tqdm(range(1, common.n_test+1)):
        rewards, wins = countAgent.runLoop()
    lps = lps_dict(countAgent.countDict)
    only = getOnlyRewards(lps)
    normalized_dict = normalize(lps)

    pprint(only)
    pprint(normalized_dict)
    # pprint(getOnlyHands(lps))

    fig = plt.figure(figsize=(8,5))
    fig.add_subplot(211)
    plt.title("normalized")
    plt.bar(*zip(*normalized_dict.items()))
    fig.add_subplot(212)
    plt.bar(*zip(*only.items()))
    plt.title("not normalized")
    plt.show()

    # print(countAgent.XVecs)
    X = np.array(countAgent.XVecs)
    Y = np.array(countAgent.YVec)

    print(X.shape)
    print(Y.shape)
    print(X)
    w = np.linalg.inv(X.T @ X) @ X.T @ Y
    
    print(w)
    d = dict()
    for i in range(14):
        if i != 0:
            d[i] = w[i] * 1000

    pprint(d)
    # fig = plt.figure(figsize=(8,5))
    # fig.add_subplot(111)
    # plt.title("Histogram")
    # plt.bar(*zip(*getOnlyHands(lps).items()))
    # plt.show()

def main():
    # finalTest(vec=True)
    batchGames(vec=True)

if __name__ == '__main__':
    main()
