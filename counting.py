from tqdm import tqdm
import learning
import simulator as sim
import basic_strategy as bs
import common
import itertools
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use( 'tkagg' )



def initCountDict(game):
    total_min_count_val = common.COUNT_MIN_VAL_DECK * game.shoe.n
    total_max_count_val = common.COUNT_MAX_VAL_DECK * game.shoe.n
    d = dict()

    for running_count in range(total_min_count_val, total_max_count_val+1):
        for decks in range(1,game.shoe.n + 1):
            d[float(running_count)/decks] = (0,0)
    return d

def normalize(d : dict):
    norm = dict()
    for key in d.keys():
        (rewards, hands) = d[key]
        if(hands != 0):
            if hands < 10000:
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
        if abs(key) < 20:
            # rounded = round(key * 2) / 2
            rounded = round(key)
            if rounded not in lps_dict:
                lps_dict[rounded] = d[key]
            else:
                (lrewards, lhands) = lps_dict[rounded]
                (rrewards, rhands) = d[key]
                lps_dict[rounded] = (lrewards + rrewards, lhands + rhands)
    return lps_dict
class CountAgent:
    def __init__(self):
        self.game = sim.Game()
        self.Q_table = bs.initBasicStrategy()
        self.countDict = initCountDict(self.game)
    
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
    def getBet(self):
        #TODO
        return 1

    def runLoop(self):
        self.game.place_bet(self.getBet())
        game_state, player_state = self.game.reset_hands()
        count = self.game.get_count()
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

        (count_rewards, touched) = self.countDict[count]
        self.countDict[count] = (count_rewards + rewards, touched + 1)
        return rewards, wins

def batchGames(countAgent: CountAgent):
    countAgent.game.shoe.rebuild()
    while countAgent.game.money > countAgent.game.minBet:
        print("Money: " + str(countAgent.game.money) + "\r") 
        countAgent.runLoop()

def finalTest():
    countAgent = CountAgent()
    for _ in tqdm(range(1, common.n_test+1)):
        rewards, wins = countAgent.runLoop()
    lps = lps_dict(countAgent.countDict)
    only = getOnlyRewards(lps)
    normalized_dict = normalize(lps)

    # pprint(only)
    # pprint(normalized_dict)
    # pprint(getOnlyHands(lps))

    fig = plt.figure(figsize=(8,5))
    fig.add_subplot(211)
    plt.title("normalized")
    plt.bar(*zip(*normalized_dict.items()))
    fig.add_subplot(212)
    plt.bar(*zip(*only.items()))
    plt.title("not normalized")
    plt.show()

    # fig = plt.figure(figsize=(8,5))
    # fig.add_subplot(111)
    # plt.title("Histogram")
    # plt.bar(*zip(*getOnlyHands(lps).items()))
    # plt.show()

def main():
    finalTest()

if __name__ == '__main__':
    main()
