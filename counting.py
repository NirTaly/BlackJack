import os
import random

from tqdm import tqdm
import learning
import simulator_2 as sim
import basic_strategy_simple as bs
import common
from collections import defaultdict
import itertools
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool


#TODO:
# [V]  XVecs only for true_count > -2 (or some other thereshold)
# []   HiLo bet if true_count>0.5

## Global Vars
lock = Lock()

decks_list = [4,5,6,7,8]
pen_list = [6/10, 2/3, 7/10, 7.5/10, 8/10, 9/10, 10/10]
spread_list = [common.spread]
learn_vec_args = [
        (4, 2.5/4, 25), (4, 3/4, 25), (4, 4/4, 25),
        (5, 3/5, 25), (5, 3.5/5, 25), (5, 4/5, 25), (5, 5/5, 25),
        (6, 4/6, 25), (6, 4.5/6, 25), (6, 5/6, 25), (6, 6/6, 25),
        (7, 5/7, 25), (7, 5.5/7, 25), (7, 6/7, 25), (7, 7/7, 25),
        (8, 6/8, 25), (8, 6.5/8, 25), (8, 7/8, 25), (8, 8/8, 25)
        ]
learn_vec_args = random.shuffle(learn_vec_args) # for the lol
##

def saveToDB(key, d, file='data/dicts.pkl'):
    with lock:
        try:
            # Try to open the file with mode 'a+' and load the data
            with open(file, 'ab+') as f:
                f.seek(0)  # Move the file pointer to the beginning
                data = pickle.load(f)
        except EOFError:
            # If the file is empty, create an empty dictionary
            data = {}

        data[key] = d
        with open(file, 'wb') as f:
            pickle.dump(data, f)

def loadFromDB(key, file='data/dicts.pkl'):
    with lock:
        try:
            # Try to open the file with mode 'a+' and load the data
            with open(file, 'ab+') as f:
                f.seek(0)  # Move the file pointer to the beginning
                data = pickle.load(f)
        except EOFError:
            # If the file is empty, create an empty dictionary
            data = {}

        # Get the value for the given key or an empty dictionary if it does not exist
        d = data.get(key, {})
        return d

def initializeDB():
    saveToDB('countDict_True', {})
    saveToDB('winrateDictVec', {})
    # saveToDB('countDict_False', {})
    # saveToDB('winrateDict', {})
    # saveToDB('countDict_False_1', {})
    # saveToDB('winrateDict1', {})


def join_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Join two dictionaries.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary that holds the join of the two dictionaries. If a key is in both dictionaries,
        it will save the sum of the values as `(a.val1 + b.val1, a.val2 + b.val2)`. If a key is only in one dictionary,
        it will save the value from that dictionary.
    """
    return {k: (dict1[k][0] + dict2[k][0], dict1[k][1] + dict2[k][1]) if k in dict1 and k in dict2 else dict1.get(k, dict2.get(k)) for k in set(dict1) | set(dict2)}

def roundCount(count):
    return round(count * 2)/2

def initCountDict(game):
    total_min_count_val = common.COUNT_MIN_VAL_DECK * game.shoe.n
    total_max_count_val = common.COUNT_MAX_VAL_DECK * game.shoe.n
    d = dict()

    for running_count in range(total_min_count_val, total_max_count_val+1):
        for i in range(0,10):
            d[round(float(running_count) + 0.1 * i, 1)] = (0, 0)
    return d

def normalize(d : dict, vec):
    norm = dict()
    max_count, min_count = 0, 0
    for count in d.keys():
        (rewards, hands) = d[count]
        if hands > common.lps_threshold:
            if count > max_count:
                max_count = count
            elif count < min_count:
                min_count = count
            norm[count] = (rewards / hands)
        else:
            norm[count] = 0
    if vec:
        common.lps_limit_max_vec = max_count
        common.lps_limit_min_vec = min_count
    else:
        common.lps_limit_max = max_count
        common.lps_limit_min = min_count
    print(f'max count = {max_count} , min count = {min_count}')

    filtered_dict = {k: v for k, v in norm.items() if min_count <= k <= max_count}
    return filtered_dict

def getOnlyRewards(vec, d : dict):
    min_count = common.lps_limit_min_vec if vec else common.lps_limit_min
    max_count = common.lps_limit_max_vec if vec else common.lps_limit_max

    return {count: rewards for count, (rewards, hands) in d.items() if (hands > 0 and min_count <= count <= max_count)}

def getOnlyHands(vec, d : dict):
    min_count = common.lps_limit_min_vec if vec else common.lps_limit_min
    max_count = common.lps_limit_max_vec if vec else common.lps_limit_max

    return {count: hands for count, (rewards, hands) in d.items() if (hands > 0 and min_count <= count <= max_count)}

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


def createLpsDict(vec=False, countDict=None):
    lps_dict = defaultdict(lambda: [0, 0])
    lps_limit = getLPSLimit(vec)

    for count, (rewards, hands) in countDict.items():
        if abs(count) < lps_limit:
            rounded = roundCount(count)
            lps_dict[rounded][0] += rewards
            lps_dict[rounded][1] += hands

    return dict(lps_dict)

class CountAgent:
    def __init__(self, needMatrixes, vec=False, spread=common.spread, penetration=common.penetration, kellyFrac=common.vec_kelly_fraction, decks=common.num_of_decks):
        self.game = sim.Game(decks, penetration)
        self.Q_table = bs.initBasicStrategy('das_sas')
        self.countDict = initCountDict(self.game)
        if needMatrixes:
            self.XVecs = np.zeros((common.n_test, 11), dtype=np.float16)
            self.YVec = np.zeros(common.n_test, dtype=np.float16)
        self.vec = vec
        self.needMatrixes = needMatrixes
        self.spread = spread
        self.kelly_frac = kellyFrac
        self.w = None

        self.initVectorW()

    def getPKLkey(self):
        return f'decks_{self.game.shoe.n}_penetration_{round(self.game.shoe.n*self.game.shoe.penetration,2)}_spread_{self.spread}'

    def initVectorW(self):
        self.w = loadFromDB(self.getPKLkey(), 'data/w_vectors.pkl')

    def handleBJ(self):
        reward, done = 0, False
        player_state, _ = self.game.sum_hands(self.game.playerCards[0])
        dealer_sum, _ = self.game.sum_hands(self.game.dealerCards)
        if player_state == 21 or dealer_sum == 21:
            reward = self.game.rewardHandler(dealer_sum, [player_state])
            if reward > 0:
                self.game.money += (1/2) * self.game.bet
                reward *= 1.5
            done = True
        return reward, done

    def getAction(self, game_state, player_state, dealer_state):
        player_state = learning.playerStateFromGameState(self.game, game_state, player_state)
        action = self.Q_table[game_state][(player_state, dealer_state)]

        if action == 'D' and not (self.game.isAfterSplit() or self.game.first_move):
            if player_state > 17:
                action = 'S'
            else:
                action = 'H'

        if action == 'X' and not self.game.first_move:
            action = 'H'

        return action

    # function that place the best bet, probably according to Kelly criterion
    def getBet(self, betMethod):
        if betMethod == 'const':
            return 1
        else:
            if self.vec:
                vector = self.game.shoe.countVec
                E = (self.w.T @ vector / 1000) / self.game.shoe.rem_decks
            else:
                count = roundCount(self.game.get_count(self.vec))
                if count < getWinrateMaxMin(max=False,vec=self.vec):
                    count = getWinrateMaxMin(max=False,vec=self.vec)
                elif count > getWinrateMaxMin(max=True,vec=self.vec):
                    count = getWinrateMaxMin(max=True,vec=self.vec)

                E = common.winrateDict[count]

            if betMethod == 'spread':
                return self.game.minBet if E <= 0 else self.game.minBet * self.spread

            elif betMethod == 'kelly':
                p = 0.5 + E
                q = 1 - p
                f = p - q

                f *= self.kelly_frac
                bet = max (self.game.minBet, int(self.game.money * f))
                return bet

    def runLoop(self, testIdx, betMethod='const', needCountDict=False):
        count = self.game.get_count(vec=self.vec)
        if self.needMatrixes:
            countVec = tuple(self.game.shoe.getNormVec())

        self.game.place_bet(self.getBet(betMethod))

        game_state, player_state = self.game.reset_hands()

        reward, done = self.handleBJ()
        for i, _ in enumerate(self.game.playerCards):
            while not done:
                action = self.getAction(game_state, player_state, self.game.dealer_state)

                game_state, player_state, reward, done = self.game.step(action)

            if self.game.isSplit:
                if self.game.playerCards[0][0] == 1 and self.game.splitAcesAndDone:
                    break
                if self.game.currHand < len(self.game.playerCards):
                    player_state, game_state = self.game.sum_hands(self.game.playerCards[self.game.currHand])
                done = False

        if self.needMatrixes and count > common.lin_reg_min_threshold:
            self.XVecs[testIdx, :] = countVec
            self.YVec[testIdx] = reward
        elif needCountDict:
            (count_rewards, touched) = self.countDict[round(count, 1)]
            self.countDict[round(count, 1)] = (count_rewards + reward, touched + 1)

        return reward


def setMaxMin(d : dict, vec):
    max_count, min_count = 0, 0
    for count in d.keys():
        expectedWinrate = d[count]
        if expectedWinrate != 0:
            if count > max_count:
                max_count = count
            elif count < min_count:
                min_count = count
    if vec:
        common.lps_limit_max_vec = roundCount(common.lps_minmax * max_count)
        common.lps_limit_min_vec = roundCount(common.lps_minmax * min_count)
    else:
        common.lps_limit_max = roundCount(common.lps_minmax * max_count)
        common.lps_limit_min = roundCount(common.lps_minmax * min_count)

    print(f'max = {roundCount(common.lps_minmax * max_count)} , min = {roundCount(common.lps_minmax * min_count)}')

def linear_reg(countAgent : CountAgent, pkl_key=None):
    print("\nStarting linaer_reg")

    X = countAgent.XVecs
    Y = countAgent.YVec

    zero_rows_indices = np.where(~X.any(axis=1))[0]
    X = np.delete(X, zero_rows_indices, axis=0)
    Y = np.delete(Y, zero_rows_indices)

    # print(X.shape)
    # print(Y.shape)
    # print("Calculating w..\n")

    model = LinearRegression(copy_X=False, n_jobs=-1).fit(X, Y)
    #model = Ridge(copy_X=False).fit(X, Y)
    w = model.coef_
    bias = model.intercept_

    #w = np.linalg.inv(X.T @ X) @ X.T @ Y

    #w[10] = w[10] / 4   # Normalize after the calculation

    #pprint(f'Bias = {bias}')
    d = dict()
    for i in range(1,11):
        d[i] = w[i] * 1000

    #pprint(d)
    #print(f'Sum of vector = {sum(d.values()) + 3*d[10]}')

    common.w = np.concatenate((np.array([bias]), 1000*np.array(w[1:])))
    #print(f'common.w = {common.w}')

    sim.cards_vec_count = d
    with open("data/w_vector.txt", "a") as f:
        pprint(pkl_key, f)
        pprint(f'Bias = {bias}', f)
        pprint(d,f)
        pprint(f'Sum = {sum(d.values()) + 3*d[10]}', f)

    if pkl_key is not None:
        saveToDB(pkl_key, common.w, 'data/w_vectors.pkl')

def count_graphs(vec, countDict):
    lps = createLpsDict(vec, countDict)
    only = getOnlyRewards(vec, lps)
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
        saveToDB('winrateDict1', normalized_dict)
        with open("data/only_rewards_hilo.txt", "w") as f:
            pprint(only, f)
        with open("data/normalized_dict_hilo.txt", "w") as f:
            pprint(normalized_dict, f)

    total_hands = sum(value[1] for value in countDict.values())

    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(f'Total Hands Trained: {total_hands / common.BILLION:.2f} Bil')

    ax1 = fig.add_subplot(211)
    ax1.set_title(f'Reward per Count\nNormalized by # of Occurrences per Count ', fontsize=10)
    ax1.set_xlabel('Count')
    ax1.set_ylabel(r'$\frac{Reward_{i}}{N_{i}}$')
    ax1.set_yticks(np.arange(-0.4, 0.5, 0.2))
    ax1.grid(True)
    ax1.bar(*zip(*normalized_dict.items()))

    plt.subplots_adjust(hspace=0.5)

    ax2 = fig.add_subplot(212)
    ax2.bar(*zip(*only.items()))
    ax2.set_title(f'Total Rewards per Count', fontsize=10)
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Rewards')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.grid(True)

    plt.savefig(f'res/CountNormalized_vec_{vec}', dpi=500)

    fig = plt.figure(figsize=(8,5))
    fig.add_subplot(111)
    plt.title("Histogram")
    plt.xlabel('Count')
    plt.ylabel('# of Occurrences')
    plt.bar(*zip(*getOnlyHands(vec, lps).items()))
    plt.grid(True)
    plt.savefig(f'res/Histogram_{vec}')


def finalTest(vec = False):
    countAgent = CountAgent(needMatrixes=False, vec=vec)
    for i in tqdm(range(common.n_test)):
        countAgent.runLoop(i, needCountDict=True)

    # last_countDict = loadFromDB(f'countDict_{vec}')
    if not vec:
        last_countDict = loadFromDB(f'countDict_False_1')
    else:
        last_countDict = loadFromDB(f'countDict_{vec}')
    new_countDict = join_dicts(countAgent.countDict, last_countDict)

    # saveToDB(f'countDict_{vec}', new_countDict)
    if not vec:
        saveToDB(f'countDict_False_1', new_countDict)
    else:
        saveToDB(f'countDict_{vec}', new_countDict)

    count_graphs(vec, new_countDict)

def singleGame(args):
    (i, vec, betMethod, decks, spread, penetration, kellyFrac) = args

    data = np.zeros(common.graphs_max_hands)
    failed = True
    countAgent = None
    while failed:
        countAgent = CountAgent(needMatrixes=False, vec=vec, spread=spread, penetration=penetration, kellyFrac=kellyFrac, decks=decks)
        hands = 0
        data[0] = countAgent.game.money
        while hands < common.graphs_max_hands:
            if countAgent.game.money < countAgent.game.minBet and betMethod != "spread":
                failed = True
                # print(f"I SUCK ASS! {(i, vec, betMethod, spread, penetration, kellyFrac)} money = {countAgent.game.money} hands = {hands}")
                break
            else:
                # print(f'{countAgent.game.money} , hand: {hands}')
                failed = False

            countAgent.runLoop(hands, betMethod, needCountDict=False)  # i=0 is redundent, just to pass something
            if hands % common.graphs_sample_rate == 0:
                data[hands] = countAgent.game.money
            hands += 1

    return (i, data, countAgent.game.total_bets / ((common.graphs_max_hands - 1) * common.min_bet))


def multyProcessBatchGames(vec=False, betMethod='kelly', decks=common.num_of_decks, spread=common.spread, penetration=common.penetration, kellyFrac=common.vec_kelly_fraction, queue=None):
    data = np.zeros((common.graphs_num_of_runs, common.graphs_max_hands))
    x_indices = np.arange(0, common.graphs_max_hands, common.graphs_sample_rate)
    
    # pool = mp.Pool(os.cpu_count())
    pool = mp.Pool(min(30, common.graphs_num_of_runs))
    avg_bet_sum = 0
    args = [(i, vec, betMethod, decks, spread, penetration, kellyFrac) for i in range(common.graphs_num_of_runs)]

    for (i, run_data, avg_bet) in tqdm(pool.imap_unordered(singleGame, args), total=common.graphs_num_of_runs):
        avg_bet_sum += avg_bet
        data[i] = run_data

    if queue:
        queue.put([vec, data[:, x_indices], avg_bet_sum])
    else:
        return data[:, x_indices], avg_bet_sum


# betMethod: 'const'/'kelly'/'spread'
def batchGamesAndFig(betMethod='kelly'):
    x_indices = np.arange(0, common.graphs_max_hands, common.graphs_sample_rate)
    hiloData, hilo_avg_bet_sum = multyProcessBatchGames(vec=False, betMethod=betMethod)
    vecData, vec_avg_bet_sum = multyProcessBatchGames(vec=True, betMethod=betMethod)

    vec_mean = np.mean(vecData, axis=0)
    vec_gmean = gmean(vecData, axis=0)
    vec_std = np.std(vecData, axis=0)

    hilo_mean = np.mean(hiloData, axis=0)
    hilo_gmean = gmean(hiloData, axis=0)
    hilo_std = np.std(hiloData, axis=0)

    if betMethod == 'spread':
        print('Vec results:')
        print(f'E[spread edge]\t=\t{(vec_mean[-1] - common.initial_money) / ((common.graphs_max_hands - 1) * common.min_bet)}')
        print(f'E[bet]\t\t=\t{vec_avg_bet_sum / common.graphs_num_of_runs}')

        print('HiLo results:')
        print(f'E[spread edge]\t=\t{(hilo_mean[-1] - common.initial_money) / ((common.graphs_max_hands - 1) * common.min_bet)}')
        print(f'E[bet]\t\t=\t{hilo_avg_bet_sum / common.graphs_num_of_runs}')

    fig, ax = plt.subplots(figsize=(8,5))
    plt.xlabel('Hands')
    plt.ylabel('Money')
    if betMethod == 'kelly':
        ax.set_yscale('log')
        ax.set_title(f"Money vs Hands\n{common.graphs_num_of_runs} games\nVec Kelly Fraction: {common.vec_kelly_fraction}\nHiLo Kelly Fraction: {common.hilo_kelly_fraction}")
        # plt.title(f"Money - Hands\n{common.graphs_num_of_runs} games\nVec Kelly Frac:{common.vec_kelly_fraction}    HiLo Kelly Frac:{common.hilo_kelly_fraction}")
        ax.plot(x_indices, vec_gmean, label='Vec Geometric Mean', color='blue', linestyle='-')
        ax.plot(x_indices, hilo_gmean, label='HiLo Geometric Mean', color='red', linestyle='-')

    # Vec subplots
    ax.plot(x_indices, vec_mean, label='Vec Mean', color='blue', linestyle='--')
    # HiLo subplots
    ax.plot(x_indices, hilo_mean, label='HiLo Mean', color='red', linestyle='--')

    if betMethod == 'spread':
        plt.ticklabel_format(style='sci', axis='y', scilimits=(5,5))
        # plt.title(f"Money - Hands\n{common.graphs_num_of_runs} games\nSpread: {common.spread}")
        ax.set_title(f"Money vs Hands\n{common.graphs_num_of_runs} games\nSpread: {common.spread}")

        vec_edge = float("{:.6f}".format(vec_mean[-1]/(common.initial_money * (common.graphs_max_hands-1))))
        hilo_edge = float("{:.6f}".format(hilo_mean[-1]/(common.initial_money * (common.graphs_max_hands-1))))
        ax.text(0.34,0.85,f'Vec Edge: {vec_edge}\nHiLo Edge: {hilo_edge}\nVec is Better By: { float("{:.6f}".format((vec_edge/hilo_edge)-1))}',transform=ax.transAxes)
    else:
        vec_g = float("{:.6f}".format(np.power(vec_mean[-1]/common.initial_money, 1/(common.graphs_max_hands-1))))
        hilo_g = float("{:.6f}".format(np.power(hilo_mean[-1]/common.initial_money, 1/(common.graphs_max_hands-1))))
        ax.text(0.34,0.85,f'Vec Growth: {vec_g}\nHiLo Growth: {hilo_g}\nVec is Better By: { float("{:.6f}".format((vec_g/hilo_g)-1))}',transform=ax.transAxes)

    ax.legend()
    ax.grid(True)
    plt.savefig(f'res/MoneyGraph_{betMethod}', dpi=500)

    # STD plots
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel('Hands')
    ax.set_ylabel('Money')
    ax.set_title(f"Mean with Standard Deviation\nComputing over {common.graphs_num_of_runs} games")

    # # Vec subplot
    ax.plot(x_indices, vec_mean, label='Vec Mean', color='blue', linestyle='--')
    # # HiLo subplot
    ax.plot(x_indices, hilo_mean, label='HiLo Mean', color='red', linestyle='--')

    if betMethod == 'spread':
        plt.ticklabel_format(style='sci', axis='y', scilimits=(5, 5))
    
    
    # ax.fill_between(x_indices, hilo_mean - 2*hilo_std, hilo_mean + 2*hilo_std, alpha=0.4)
    
    if betMethod == 'kelly':
        ax.set_yscale('log')
        ax.fill_between(x_indices, vec_mean, vec_mean + vec_std, alpha=0.2)
        ax.fill_between(x_indices, hilo_mean, hilo_mean + hilo_std, alpha=0.2)

        vec_g = float("{:.6f}".format(np.power(vec_mean[-1] / common.initial_money, 1 / (common.graphs_max_hands - 1))))
        hilo_g = float("{:.6f}".format(np.power(hilo_mean[-1] / common.initial_money, 1 / (common.graphs_max_hands - 1))))
        ax.text(0.34, 0.85,f'Vec Growth: {vec_g}\nHiLo Growth: {hilo_g}\nVec is Better By: {float("{:.6f}".format((vec_g / hilo_g) - 1))}',transform=ax.transAxes)
    else:
        ax.fill_between(x_indices, vec_mean - vec_std, vec_mean + vec_std, alpha=0.2)
        ax.fill_between(x_indices, hilo_mean - hilo_std, hilo_mean + hilo_std, alpha=0.2)

        vec_edge = float("{:.6f}".format(vec_mean[-1] / (common.initial_money * (common.graphs_max_hands - 1))))
        hilo_edge = float("{:.6f}".format(hilo_mean[-1] / (common.initial_money * (common.graphs_max_hands - 1))))
        ax.text(0.34, 0.85, f'Vec Edge: {vec_edge}\nHiLo Edge: {hilo_edge}\nVec is Better By: {float("{:.6f}".format((vec_edge / hilo_edge) - 1))}', transform=ax.transAxes)

    ax.legend()
    ax.grid(True)
    plt.savefig(f'res/STDGraph_{betMethod}', dpi=500)

def MPBatchGamesWithArgs(vec, betMethod, penetration=common.penetration, kellyFrac=common.vec_kelly_fraction, spread=common.spread, queue=None, x=None):
    data,_ = multyProcessBatchGames(vec=vec, betMethod=betMethod, penetration=penetration, kellyFrac=kellyFrac, spread=spread)

    # Compute statistics for vecData
    data_mean = np.mean(data, axis=0)
    data_gmean = gmean(data, axis=0)
    data_std = np.std(data, axis=0)

    # Put the results in the queue
    queue.put([data_mean[-1], data_gmean[-1], data_std[-1], x])

def handleMP(testedArg, x_indices, betMethod='kelly'):
    # Initialize empty dictionaries for storing results
    vecDatas_mean = dict()
    vecDatas_gmean = dict()
    vecDatas_std = dict()
    hiloDatas_mean = dict()
    hiloDatas_gmean = dict()
    hiloDatas_std = dict()

    # Create an empty list for storing processes
    results = []

    # Create a queue for communication
    queue = mp.Queue()

    # Create two processes, one for vec=True and one for vec=False
    p1 = mp.Process(target=handleMPArgs, args=(testedArg, x_indices['vec'] , betMethod, True , queue))
    p2 = mp.Process(target=handleMPArgs, args=(testedArg, x_indices['hilo'], betMethod, False, queue))

    # Start the processes and append them to the list
    p1.start()
    p2.start()
    results.append(p1)
    results.append(p2)

    # Wait for both processes to finish and get their results from the queue
    for p in results:
        p.join()
        returned_values = queue.get()
        vec = returned_values[3]
        if vec:
            # Store the results in the dictionaries for vec=True
            vecDatas_mean = returned_values[0]
            vecDatas_gmean = returned_values[1]
            vecDatas_std = returned_values[2]
        else:
            # Store the results in the dictionaries for vec=False
            hiloDatas_mean = returned_values[0]
            hiloDatas_gmean = returned_values[1]
            hiloDatas_std = returned_values[2]

    # Return the dictionaries of results
    return [vecDatas_mean, vecDatas_gmean, vecDatas_std, hiloDatas_mean, hiloDatas_gmean, hiloDatas_std]

def handleMPArgs(testedArg, x_indices, betMethod='kelly', vec=True, queue=None):
    # Initialize empty dictionaries for storing results
    datas_mean = dict()
    datas_gmean = dict()
    datas_std = dict()

    # Create an empty list for storing processes
    results = []

    # Create a queue for communication
    queue2 = mp.Queue()

    # Set the common arguments for the functions
    args = [vec, betMethod, common.penetration, common.vec_kelly_fraction if vec else common.hilo_kelly_fraction, common.spread, queue2]

    # Loop over the x_indices for vec=True
    for x in x_indices:
        # Update the argument that is being tested
        if testedArg == 'penetration':
            args[2] = x
        elif testedArg == 'kelly_fraction':
            args[3] = x
        elif testedArg == 'spread':
            args[4] = x
        else:
            raise Exception("WRONG testedArg")

        # Create a process with MPBatchGamesWithArgs function and arguments
        p = mp.Process(target=MPBatchGamesWithArgs, args=tuple(args + [x]))

        # Start the process and append it to the list
        p.start()
        results.append(p)

    # Wait for all processes to finish and get their results from the queue2
    for p in results:
        p.join()
        returned_values = queue2.get()
        x = returned_values[3]
        datas_mean[x] = returned_values[0]
        datas_gmean[x] = returned_values[1]
        datas_std[x] = returned_values[2]

    # Put the dictionaries of results and the vec value in the queue
    queue.put([datas_mean, datas_gmean, datas_std, vec])

#TODO change function to handle variable num of decks
def batchGamesAndPenetrate(betMethod='kelly'):
    a,b = common.penetration_sample_rate.split('/')
    pen_rate = float(a) / float(b)
    x_indices = dict()
    x_indices['vec'] = np.arange(1,common.num_of_decks*pen_rate,-pen_rate)
    x_indices['hilo'] = np.arange(1,common.num_of_decks*pen_rate,-pen_rate)

    results = handleMP(testedArg='penetration', x_indices=x_indices, betMethod=betMethod)
    vecDatas_mean    = {k*common.num_of_decks:v for k,v in results[0].items()}
    vecDatas_gmean   = {k*common.num_of_decks:v for k,v in results[1].items()}
    vecDatas_std     = {k*common.num_of_decks:v for k,v in results[2].items()}
    hiloDatas_mean   = {k*common.num_of_decks:v for k,v in results[3].items()}
    hiloDatas_gmean  = {k*common.num_of_decks:v for k,v in results[4].items()}
    hiloDatas_std    = {k*common.num_of_decks:v for k,v in results[5].items()}

    best_penetration = round(max(vecDatas_gmean, key=vecDatas_gmean.get), 2)
    print(f'Best Penetration: {best_penetration}')
    common.penetration = best_penetration

    fig, ax = plt.subplots()
    plt.title(f'Money - Penetration\nPenetration Sample Rate = {common.penetration_sample_rate}\nRunning {common.graphs_num_of_runs} games each')
    plt.xlabel(f'# of Played Decks out of {common.num_of_decks}')
    plt.xticks(np.arange(0, common.num_of_decks+0.5, 0.5))
    plt.ylabel('Money')
    if betMethod == 'spread':
        plt.ticklabel_format(style='sci', axis='y', scilimits=(5, 5))
    elif betMethod == 'kelly':
        ax.set_yscale('log')

    # Vec subplots
    ax.plot(*zip(*sorted(vecDatas_mean.items())), label='Vec Mean', color='blue', linestyle='--')
    ax.plot(*zip(*sorted(vecDatas_gmean.items())), label='Vec Geo Mean', color='blue', linestyle='-')
    # HiLo subplots
    ax.plot(*zip(*sorted(hiloDatas_mean.items())), label='HiLo Mean', color='red', linestyle='--')
    ax.plot(*zip(*sorted(hiloDatas_gmean.items())), label='HiLo Geo Mean', color='red', linestyle='-')

    plt.legend()
    plt.grid(True)
    plt.savefig(f'res/MoneyPerPenetration_{betMethod}', dpi=500)


def kellyFractionGraph():
    print("\nStarting kellyFractionGraph")
    x_indices = dict()
    x_indices['vec']  = np.arange(0.3,0.7,common.kelly_frac_sample_rate)
    x_indices['hilo'] = np.arange(0.7,1.3,common.kelly_frac_sample_rate)

    results = handleMP(testedArg='kelly_fraction', x_indices=x_indices, betMethod='kelly')
    # vecDatas_mean = results[0]
    vecDatas_gmean = results[1]
    # vecDatas_std = results[2]
    # hiloDatas_mean = results[3]
    hiloDatas_gmean = results[4]
    # hiloDatas_std = results[5]

    best_vec_kelly_frac = round(max(vecDatas_gmean, key=vecDatas_gmean.get),2)
    best_hilo_kelly_frac = round(max(hiloDatas_gmean, key=hiloDatas_gmean.get),2)
    print(f'Best Vec Kelly Fraction: {best_vec_kelly_frac}')
    print(f'Best HiLo Kelly Fraction: {best_hilo_kelly_frac}')
    common.vec_kelly_fraction = best_vec_kelly_frac
    common.hilo_kelly_fraction = best_hilo_kelly_frac

    fig, ax = plt.subplots()
    plt.title(f'Money - Kelly Fraction\nRunning {common.graphs_num_of_runs} games each')
    plt.xlabel(f'Kelly Fraction')
    plt.ylabel('Money')
    ax.set_yscale('log')

    # Vec subplots
    # ax.plot(*zip(*sorted(vecDatas_mean.items())), label='Vec Mean', color='blue', linestyle='--')
    ax.plot(*zip(*sorted(vecDatas_gmean.items())), label='Vec Geo Mean', color='blue', linestyle='-')
    # HiLo subplots
    # ax.plot(*zip(*sorted(hiloDatas_mean.items())), label='HiLo Mean', color='red', linestyle='--')
    ax.plot(*zip(*sorted(hiloDatas_gmean.items())), label='HiLo Geo Mean', color='red', linestyle='-')

    plt.legend()
    plt.grid(True)
    plt.savefig(f'res/MoneyPerKellyFrac', dpi=500)

def spreadGraph():
    print("\nStarting spreadGraph")
    x_indices = dict()
    x_indices['vec'] = np.arange(10,100,common.spread_sample_rate)
    x_indices['hilo'] = np.arange(10,100,common.spread_sample_rate)

    results = handleMP(testedArg='spread', x_indices=x_indices, betMethod='spread')
    vecDatas_mean = results[0]
    # vecDatas_gmean = results[1]
    vecDatas_std = results[2]
    hiloDatas_mean = results[3]
    # hiloDatas_gmean = results[4]
    hiloDatas_std = results[5]

    fig, ax = plt.subplots()
    plt.title(f'Money - Spread\nRunning {common.graphs_num_of_runs} games each')
    plt.xlabel(f'Spread Value')
    plt.ylabel('Money')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(5, 5))

    # Vec subplots
    x_vec, y_vec = zip(*sorted(vecDatas_mean.items()))
    ax.plot(x_vec, y_vec, label='Vec Mean', color='blue', linestyle='--')

    x_vec_std, y_vec_std = zip(*sorted(vecDatas_std.items()))
    ax.fill_between(x_vec_std, np.array(y_vec) - np.array(y_vec_std), np.array(y_vec) + np.array(y_vec_std), alpha=0.2)

    # HiLo subplots
    x_hilo, y_hilo = zip(*sorted(hiloDatas_mean.items()))
    ax.plot(x_hilo, y_hilo, label='HiLo Mean', color='red', linestyle='--')

    x_hilo_std, y_hilo_std = zip(*sorted(hiloDatas_std.items()))
    ax.fill_between(x_hilo_std, np.array(y_hilo) - np.array(y_hilo_std), np.array(y_hilo) + np.array(y_hilo_std),
                    alpha=0.2)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'res/MoneyPerSpread', dpi=500)

def learnVecPerArg():
    pool = mp.Pool(2)

#    args = []
#    for decks in decks_list:
#        for penetration in pen_list:
#            for spread in spread_list:
#                pkl_key = f'decks_{decks}_penetration_{round(decks*penetration,2)}_spread_{spread}'
#                print(pkl_key)
#                args.append((decks,penetration,spread, pkl_key))
    for i, (decks, penetration, spread) in enumerate(learn_vec_args):
        pkl_key = f'decks_{decks}_penetration_{round(decks*penetration,2)}_spread_{spread}'
        learn_vec_args[i] = (decks, penetration, spread, pkl_key)

    for _ in tqdm(pool.imap_unordered(run_create_vec, learn_vec_args), total=len(learn_vec_args)):
        pass

def findBestArgs():
    vec_advg_dict = dict()
    pool = ThreadPool(5)

    for i, (decks, penetration, spread) in enumerate(learn_vec_args):
        pkl_key = f'decks_{decks}_penetration_{round(decks * penetration, 2)}_spread_{spread}'
        learn_vec_args[i] = (decks, penetration, spread, pkl_key)

    for (pkl_key, vec_advg, vec_mean, vec_std, hilo_mean, hilo_std) in tqdm(pool.imap_unordered(runBatchAndGetEdge, learn_vec_args), total=len(learn_vec_args)):
        vec_advg_dict[pkl_key] = vec_advg
        print(f'{pkl_key}\t\t{vec_advg}')

        spreadPlot(vec_mean=vec_mean, vec_std=vec_std, hilo_mean=hilo_mean, hilo_std=hilo_std, file=f'spread_graphs/STDSpreadGraph_{pkl_key.replace(".", "_")}')

    pprint(vec_advg_dict)
    print(f'Max Vec advg:\t{max(vec_advg_dict.values())}')
    best_pkl_key = max(vec_advg_dict, key=vec_advg_dict.get)
    print(f'{best_pkl_key}')


def runBatchAndGetEdge(args):
    (decks, penetration, spread, pkl_key) = args
    vecData, vec_avg_bet_sum = multyProcessBatchGames(vec=True, betMethod='spread', decks=decks, penetration=penetration, spread=spread)
    hiloData, hilo_avg_bet_sum = multyProcessBatchGames(vec=False, betMethod='spread', decks=decks, penetration=penetration, spread=spread)

    vec_mean = np.mean(vecData, axis=0)
    vec_std = np.std(vecData, axis=0)
    hilo_mean = np.mean(hiloData, axis=0)
    hilo_std = np.std(hiloData, axis=0)

    vec_edge = float("{:.6f}".format(vec_mean[-1] / (common.initial_money * (common.graphs_max_hands - 1))))
    hilo_edge = float("{:.6f}".format(hilo_mean[-1] / (common.initial_money * (common.graphs_max_hands - 1))))

    vec_advg = round(vec_edge / hilo_edge - 1,3)

    return (pkl_key, vec_advg, vec_mean, vec_std, hilo_mean, hilo_std)

def run_create_vec(args=None):
    print("\nStarting run_create_vec")
    (decks, penetration, spread, pkl_key) = (common.num_of_decks, common.penetration, common.spread, None)

    if args is not None:
        (decks,penetration,spread, pkl_key) = args

    countAgent = CountAgent(needMatrixes=True, vec=True, spread=spread, penetration=penetration, decks=decks)
    for i in tqdm(range(common.n_test)):
        countAgent.runLoop(i)

    linear_reg(countAgent, pkl_key)

    if pkl_key is not None:
        print(f'{pkl_key}\t\tDone!')

def multyprocessFinalTest():
    print("\nStarting multyprocessFinalTest")
    mp_pool = mp.Pool(2)
    vecs = [True, False]
    mp_pool.map(finalTest, vecs)
    mp_pool.close()
    mp_pool.join()


def multyprocessBatchAndFig():
    print("\nStarting multyprocessBatchAndFig")
    p1 = mp.Process(target=batchGamesAndFig, args=('spread',))
    p2 = mp.Process(target=batchGamesAndFig, args=('kelly',))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


def multyprocessBatchGamesAndPenetrate():
    print("\nStarting multyprocessBatchGamesAndPenetrate")
    p1 = mp.Process(target=batchGamesAndPenetrate, args=('spread',))
#    p2 = mp.Process(target=batchGamesAndPenetrate, args=('kelly',))
    p1.start()
#    p2.start()
    p1.join()
#    p2.join()

def spreadPlot(vec_mean, vec_std, hilo_mean, hilo_std, file):
    x_indices = np.arange(0, common.graphs_max_hands, common.graphs_sample_rate)
    # STD plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel('Hands')
    ax.set_ylabel('Money')
    ax.set_title(f"Mean with Standard Deviation\nComputing over {common.graphs_num_of_runs} games")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(5, 5))

    # # Vec subplot
    ax.plot(x_indices, vec_mean, label='Vec Mean', color='blue', linestyle='--')
    # # HiLo subplot
    ax.plot(x_indices, hilo_mean, label='HiLo Mean', color='red', linestyle='--')

    ax.fill_between(x_indices, vec_mean - vec_std, vec_mean + vec_std, alpha=0.2)
    ax.fill_between(x_indices, hilo_mean - hilo_std, hilo_mean + hilo_std, alpha=0.2)

    vec_edge = float("{:.6f}".format(vec_mean[-1] / (common.initial_money * (common.graphs_max_hands - 1))))
    hilo_edge = float("{:.6f}".format(hilo_mean[-1] / (common.initial_money * (common.graphs_max_hands - 1))))
    ax.text(0.34, 0.85, f'Vec Edge: {vec_edge}\nHiLo Edge: {hilo_edge}\nVec is Better By: {float("{:.6f}".format((vec_edge / hilo_edge) - 1))}', transform=ax.transAxes)

    ax.legend()
    ax.grid(True)
    plt.savefig(file, dpi=500)

def main():
    common.winrateDictVec = loadFromDB('winrateDictVec')
    common.winrateDict = loadFromDB('winrateDict1')
    setMaxMin(common.winrateDictVec, vec=True)
    setMaxMin(common.winrateDict, vec=False)
    
    # initializeDB()
    # run_create_vec()
    # multyprocessFinalTest()
    kellyFractionGraph()
    multyprocessBatchGamesAndPenetrate()
    multyprocessBatchAndFig()
    spreadGraph()

    # learnVecPerArg()
    # findBestArgs()

    # singleGame((0,False,'spread', common.num_of_decks, common.spread, common.penetration, common.vec_kelly_fraction))

if __name__ == '__main__':
    main()
