from tqdm import tqdm
from tqdm.contrib.itertools import product
import learning
import simulator as sim
import basic_strategy as bs
import common
import itertools
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import json
# import torch
# import cupy as cp
# matplotlib.use( 'tkagg' ) # we do savefig now, not needed

alreadyRunLoop = False
def roundCount(count, vec=False):
    count  = round(count * 2)/2 #if vec == False else count
    return count

def initCountDict(game):
    total_min_count_val = common.COUNT_MIN_VAL_DECK * game.shoe.n
    total_max_count_val = common.COUNT_MAX_VAL_DECK * game.shoe.n
    d = dict()

    for running_count in range(total_min_count_val, total_max_count_val+1):
        for i in range(0,10):
            d[round(float(running_count) + 0.1 * i,1)] = (0,0)
    return d

def normalize(d : dict, vec):
    norm = dict()
    max, min = 0, 0
    for key in d.keys():
        (rewards, hands) = d[key]
        # if(hands != 0):
        if hands > common.lps_threshold:
            if key > max:
                max = key
            elif key < min:
                min = key
            norm[key] = (rewards / hands) # / 2
        else:
            norm[key] = 0
    if vec:
        common.lps_limit_max_vec = max
        common.lps_limit_min_vec = min
    else:
        common.lps_limit_max = max
        common.lps_limit_min = min
    print(f'max = {max} , min = {min}')

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

def getLPSLimit(vec=False):
    return common.lps_limit_vec if vec else common.lps_limit

def getWinrateMaxMin(max, vec):
    if not vec:
        if max:
            return common.lps_limit_max
        else:
            return common.lps_limit_min
    else:
        if max:
            return common.lps_limit_max_vec - 1 #-1 for safety
        else:
            return common.lps_limit_min_vec + 1 #same


def createLpsDict(vec=False):
    d = loadFromDB('countDict')    
    lps_dict = dict()
    for key in d.keys():
        if abs(key) < getLPSLimit(vec):
            rounded = roundCount(key)
            if rounded not in lps_dict:
                lps_dict[rounded] = d[key]
            else:
                (lrewards, lhands) = lps_dict[rounded]
                (rrewards, rhands) = d[key]
                lps_dict[rounded] = (lrewards + rrewards, lhands + rhands)
    return lps_dict
class CountAgent:
    def __init__(self,needMatrixes, vec=False):
        self.game = sim.Game()
        self.Q_table = bs.initBasicStrategy()
        self.countDict = initCountDict(self.game)
        if needMatrixes:
            self.XVecs = np.zeros((common.n_test,14))
            self.YVec = np.zeros(common.n_test)
        self.vec = vec
        self.needMatrixes = needMatrixes

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
        count = roundCount(self.game.get_count(vec), vec)
        if count < getWinrateMaxMin(max=False,vec=vec):
            count = getWinrateMaxMin(max=False,vec=vec)
        elif count > getWinrateMaxMin(max=True,vec=vec):
            count = getWinrateMaxMin(max=True,vec=vec)
        if vec:
            p = 0.5 + common.winrateDictVec[count]
        else:
            p = 0.5 + common.winrateDict[count]
        q = 1 - p
        bet = max (self.game.minBet, int(self.game.money * (p - q)))
        return bet

    def runLoop(self,testIdx,kellyBet=False):
        count = self.game.get_count(vec=self.vec)
        countVec = tuple(self.game.shoe.getNormVec())
        game_state, player_state = self.game.reset_hands()
        if kellyBet:
            self.game.place_bet(self.getBet(self.vec))
        else:
            self.game.place_bet(1)
        # print(f"hand = {testIdx}")
        # print(f"count = {count}")
        # print(f"bet = {self.getBet(self.vec)}")
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
        if self.needMatrixes:
            self.XVecs[testIdx,:] = countVec
            self.YVec[testIdx] = rewards

        return rewards, wins

def batchGames(vec=False, loadWinRateFromDB=False):
    # hands = 0
    print("Starting batchGames")
    data = np.zeros((common.graphs_num_of_runs, common.graphs_max_hands))
    x_indices = np.arange(0,common.graphs_max_hands, common.graphs_sample_rate)

    if loadWinRateFromDB:
        if vec:
            common.winrateDictVec = loadFromDB('winrateDictVec')
        else:
            common.winrateDict = loadFromDB('winrateDict')

    if vec:
        setMaxMin(common.winrateDictVec, vec)
    else:
        setMaxMin(common.winrateDict, vec)

    for i in tqdm(range(common.graphs_num_of_runs)):
        not_failed = True
        while(not_failed):
            countAgent = CountAgent(needMatrixes=False, vec=vec)
            hands = 0
            data[i][hands] = countAgent.game.money
            while countAgent.game.money >= countAgent.game.minBet and hands < common.graphs_max_hands:
                #print("Money: " + str(countAgent.game.money) + f" , Hands = {hands}")#, end='\r') 
                countAgent.runLoop(hands,kellyBet=True) # i=0 is redundent, just to pass something
                if hands % common.graphs_sample_rate == 0:
                    data[i][hands] = countAgent.game.money
                    #print("Money: " + str(countAgent.game.money) + f" , Hands = {hands}")
                hands +=1

                if countAgent.game.money < countAgent.game.minBet:
                    print(f"This Sucks! I LOST ALL MY MONEY!! RUNNING AGAIN the {i}th run")
                    # i-=1 shouldnt be i-=1 because the i did not inscrease yet, we are inside a while
                    not_failed = True
                else:
                    not_failed = False

    data = data[:,x_indices]

    avg = np.mean(data, axis=0)
    percentile = np.percentile(data, 50, axis=0)
    geo_mean = gmean(data, axis=0)

    fig, ax = plt.subplots()
    plt.xlabel('Hands')
    plt.ylabel('Money')
    game_setting = 'Vec' if vec else 'HiLo'
    plt.title(f"Money - Hands\nRunning {common.graphs_num_of_runs} hands\n{game_setting} - n_test={common.n_test//common.MILLION}Mil")
    ax.set_yscale('log')
    plt.plot(x_indices, avg,label=f'Avg')
    plt.plot(x_indices, percentile,label=f'Median')
    plt.plot(x_indices, geo_mean,label=f'Geo Avg')

    plt.legend()
    plt.savefig(f'res/MoneyGraph_vec_{vec}',dpi=500)

def setMaxMin(d : dict, vec):
    max, min = 0, 0
    for key in d.keys():
        expectedWinrate = d[key]
        # if(hands != 0):
        if expectedWinrate != 0:
            if key > max:
                max = key
            elif key < min:
                min = key
    if vec:
        common.lps_limit_max_vec = max
        common.lps_limit_min_vec = min
    else:
        common.lps_limit_max = max
        common.lps_limit_min = min

    print(f'max = {max} , min = {min}')

def linear_reg(countAgent : CountAgent):
    print("\nStarting linaer_reg")
    # sortedVecs = sorted(countAgent.XVecs)

    # firstOcc = 0
    # sum = 0
    # XVecs = []
    # YVec = []
    # XVecs.append(sortedVecs[0][0])
    # for runner, vecTuple in enumerate(tqdm(sortedVecs)):
    #     if vecTuple[0] == XVecs[-1]:
    #         sum += vecTuple[1]
    #     else:
    #         YVec.append(sum/(runner-firstOcc))
    #         firstOcc = runner
    #         sum = vecTuple[1]
    #         XVecs.append(vecTuple[0])

    # # Handle last item.
    # YVec.append(sum/(len(sortedVecs[1])-firstOcc))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    X = countAgent.XVecs
    Y = countAgent.YVec

    print(X.shape)
    print(Y.shape)
    print(X)
    print("Calculating w..\n")
    w = np.linalg.inv(X.T @ X) @ X.T @ Y

    #print(w)
    print(f'Bias = {w[0]}')
    d = dict()
    for i in range(1,14):
        d[i] = w[i] * 1000

    pprint(d)
    
    print(f'Sum of vector = {sum(d.values())}')
    sim.cards_vec_count = d
    with open("data/w_vector.txt", "w") as f:
        # Writing data to a file
        pprint(d,f)

def count_graphs(vec):
    lps = createLpsDict(vec)
    only = getOnlyRewards(lps)
    normalized_dict = normalize(lps, vec)

    if vec:
        common.winrateDictVec = normalized_dict
        saveToDB('winrateDictVec', normalized_dict)
        with open("data/only_rewards_vec.txt", "w") as f:
            pprint(only, f)
        with open("data/normalized_dict_vec.txt", "w") as f:
            pprint(normalized_dict, f)
    else:
        common.winrateDict = normalized_dict
        saveToDB('winrateDict', normalized_dict)
        with open("data/only_rewards_hilo.txt", "w") as f:
            pprint(only, f)
        with open("data/normalized_dict_hilo.txt", "w") as f:
            pprint(normalized_dict, f)
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

    fig = plt.gcf()
    plt.savefig(f'res/CountNormalized_vec_{vec}',dpi=500)
    #plt.show()


    # fig = plt.figure(figsize=(8,5))
    # fig.add_subplot(111)
    # plt.title("Histogram")
    # plt.bar(*zip(*getOnlyHands(lps).items()))
    # plt.show()

def finalTest(vec = False):
    print("\nStarting finalTest")
    countAgent = CountAgent(needMatrixes=False,vec=vec)
    for i in tqdm(range(common.n_test)):
        countAgent.runLoop(i)

    saveToDB('countDict', countAgent.countDict)

    count_graphs(vec)

def saveToDB(key, d):
    with open('data/dicts.json', "r+") as f:
        data = json.load(f)
        if data is None:
            data = dict()
        data[key] = d
        f.seek(0)
        json.dump(data,f, indent=2)
def loadFromDB(key):
    with open('data/dicts.json', 'rb') as f:
        data = json.load(f)
        d = {float(k): v for k,v in data[key].items()}
        return d
def initializeDB():
    with open('data/dicts.json', 'w') as f:
        json.dump({}, f)

def run_create_vec():
    print("\nStarting run_create_vec")
    countAgent = CountAgent(needMatrixes=True, vec=True)
    for i in tqdm(range(common.n_test)):
        countAgent.runLoop(i)

    linear_reg(countAgent)
    # saveToDB('countDict', countAgent.countDict) shouldnt be here, moved to final test, using the right w vector.

def main():
    initializeDB()
    run_create_vec()
    finalTest(vec=True)
    batchGames(vec=True,loadWinRateFromDB=True) # Vectorized
    # batchGames(vec=False,loadWinRateFromDB=True) # HiLo

if __name__ == '__main__':
    main()
