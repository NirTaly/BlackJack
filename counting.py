import os

from tqdm import tqdm
import learning
import simulator as sim
import basic_strategy as bs
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

def saveToDB(key, d):
    with open('data/dicts.pkl', 'rb') as f:
        data = pickle.load(f)
    data[key] = d
    with open('data/dicts.pkl', 'wb') as f:
        pickle.dump(data, f)
def loadFromDB(key):
    with open('data/dicts.pkl', 'rb') as f:
        data = pickle.load(f)
        d = data[key]
        return d
def initializeDB():
    saveToDB('countDict_True', {})
    saveToDB('countDict_False', {})
    saveToDB('winrateDictVec', {})
    saveToDB('winrateDict', {})


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
            d[round(float(running_count) + 0.1 * i, 1)] = (0,0)
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
    # only = dict()
    # for count in d.keys():
    #     (rewards, hands) = d[count]
    #     if hands > 0:
    #         only[count] = rewards
    # return only
    min_count = common.lps_limit_min_vec if vec else common.lps_limit_min
    max_count = common.lps_limit_max_vec if vec else common.lps_limit_max

    return {count: rewards for count, (rewards, hands) in d.items() if (hands > 0 and min_count <= count <= max_count)}

def getOnlyHands(vec, d : dict):
    # only = dict()
    # for count in d.keys():
    #     (_, hands) = d[count]
    #     if hands > 0:
    #         only[count] = hands
    # return only
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
    def __init__(self, needMatrixes, vec=False, spread=common.spread, penetration=common.penetration, kellyFrac=common.kelly_fraction):
        self.game = sim.Game()
        self.Q_table = bs.initBasicStrategy()
        self.countDict = initCountDict(self.game)
        if needMatrixes:
            self.XVecs = np.zeros((common.n_test,11))
            self.YVec = np.zeros(common.n_test)
        self.vec = vec
        self.needMatrixes = needMatrixes
        self.spread = spread
        self.kelly_frac = kellyFrac

        self.game.shoe.penetration = penetration

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
        if self.game.first_move:
            player_state = learning.playerStateFromGameState(self.game, game_state, player_state)
            return max(self.Q_table[game_state][(player_state, dealer_state)],
                       key=self.Q_table[game_state][(player_state, dealer_state)].get)

        else:
            return max(dict(itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)),
                       key=dict(
                           itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)).get)

    # function that place the best bet, probably according to Kelly criterion
    def getBet(self, betMethod):
        if betMethod == 'const':
            return 1
        else:
            count = roundCount(self.game.get_count(self.vec))
            if count < getWinrateMaxMin(max=False,vec=self.vec):
                count = getWinrateMaxMin(max=False,vec=self.vec)
            elif count > getWinrateMaxMin(max=True,vec=self.vec):
                count = getWinrateMaxMin(max=True,vec=self.vec)

            if self.vec:
                # E = common.winrateDictVec[count]
                vector = self.game.shoe.countVec
                E = (common.w.T @ vector / 1000) / self.game.shoe.rem_decks
            else:
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

    def runLoop(self,testIdx,betMethod='const', needCountDict=False):
        if self.needMatrixes:
            countVec = tuple(self.game.shoe.getNormVec())
        else:
            count = self.game.get_count(vec=self.vec)

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
                player_state, game_state = self.game.sum_hands(self.game.playerCards[self.game.currHand])
                done = False
            else:
                rewards = reward

        if self.game.isSplit:
            rewards = sum(reward)
            wins = sum(1 for w in reward if w > 0)
        else:
            wins = 1 if (reward > 0) else 0

        if self.needMatrixes:
            self.XVecs[testIdx, :] = countVec
            self.YVec[testIdx] = rewards
        elif needCountDict:
            (count_rewards, touched) = self.countDict[round(count, 1)]
            self.countDict[round(count, 1)] = (count_rewards + rewards, touched + 1)

        return rewards, wins


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
        common.lps_limit_max_vec = max_count
        common.lps_limit_min_vec = min_count
    else:
        common.lps_limit_max = max_count
        common.lps_limit_min = min_count

    # print(f'max = {max_count} , min = {min_count}')

def linear_reg(countAgent : CountAgent):
    print("\nStarting linaer_reg")

    X = countAgent.XVecs
    Y = countAgent.YVec

    print(X.shape)
    print(Y.shape)
    print(X)
    print("Calculating w..\n")

    #model = LinearRegression(copy_X=False, n_jobs=-1).fit(X,Y)
    model = Ridge(copy_X=False).fit(X,Y)
    w = model.coef_
    bias = model.intercept_

    #w = np.linalg.inv(X.T @ X) @ X.T @ Y

    w[10] = w[10] / 4   # Normalize after the calculation

    print(f'Bias = {bias}')
    d = dict()
    for i in range(1,11):
        d[i] = w[i] * 1000

    pprint(d)

    print(f'Sum of vector = {sum(d.values()) + 3*d[10]}')
    sim.cards_vec_count = d
    with open("data/w_vector.txt", "w") as f:
        # Writing data to a file
        pprint(d,f)


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
        saveToDB('winrateDict', normalized_dict)
        with open("data/only_rewards_hilo.txt", "w") as f:
            pprint(only, f)
        with open("data/normalized_dict_hilo.txt", "w") as f:
            pprint(normalized_dict, f)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(211)
    plt.title(f'Reward per Count\nNormalized by # of Occurrences per Count ')
    plt.xlabel('Count')
    plt.ylabel(r'$\frac{Reward_{i}}{N_{i}}$')
    plt.yticks(np.arange(-0.4, 0.5, 0.2))
    plt.grid(True)
    plt.bar(*zip(*normalized_dict.items()))

    plt.subplots_adjust(hspace = 0.5)

    fig.add_subplot(212)
    plt.bar(*zip(*only.items()))
    plt.title("Total Rewards - Count")
    plt.xlabel('Count')
    plt.ylabel('Rewards')
    plt.grid(True)
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
    print("\nStarting finalTest")
    countAgent = CountAgent(needMatrixes=False,vec=vec)
    for i in tqdm(range(common.n_test)):
        countAgent.runLoop(i,needCountDict=True)

    last_countDict = loadFromDB(f'countDict_{vec}')
    new_countDict = join_dicts(countAgent.countDict, last_countDict)

    saveToDB(f'countDict_{vec}', new_countDict)

    count_graphs(vec, new_countDict)

def singleGame(args):
    (i, vec, betMethod, spread, penetration, kellyFrac) = args

    data = np.zeros(common.graphs_max_hands)
    failed = True
    countAgent = None
    while failed:
        countAgent = CountAgent(needMatrixes=False, vec=vec, spread=spread, penetration=penetration, kellyFrac=kellyFrac)
        hands = 0
        data[0] = countAgent.game.money
        while hands < common.graphs_max_hands:
            if countAgent.game.money < countAgent.game.minBet and betMethod != "spread":
                failed = True
                # print(f"I SUCK ASS! {(i, vec, betMethod, spread, penetration, kellyFrac)} money = {countAgent.game.money} hands = {hands}")
                break
            else:
                # print(countAgent.game.money)
                failed = False

            countAgent.runLoop(hands, betMethod, needCountDict=False)  # i=0 is redundent, just to pass something
            if hands % common.graphs_sample_rate == 0:
                data[hands] = countAgent.game.money
            hands += 1

    return (i, data, countAgent.game.total_bets / ((common.graphs_max_hands - 1) * common.min_bet))


def multyProcessBatchGames(vec=False, betMethod='kelly', spread=common.spread, penetration=common.penetration, kellyFrac=common.kelly_fraction, queue=None):
    data = np.zeros((common.graphs_num_of_runs, common.graphs_max_hands))
    x_indices = np.arange(0, common.graphs_max_hands, common.graphs_sample_rate)
    
    # pool = mp.Pool(os.cpu_count())
    pool = mp.Pool(min(20, common.graphs_num_of_runs))
    avg_bet_sum = 0
    args = [(i, vec, betMethod, spread, penetration, kellyFrac) for i in range(common.graphs_num_of_runs)]

    for (i, run_data, avg_bet) in tqdm(pool.imap_unordered(singleGame, args), total=common.graphs_num_of_runs):
        avg_bet_sum += avg_bet
        data[i] = run_data

    if queue:
        queue.put([vec, data[:, x_indices], avg_bet_sum])
        return
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

    fig, ax = plt.subplots()
    plt.title(f"Money - Hands\nRunning {common.graphs_num_of_runs} games\nn_test = {common.n_test // common.MILLION}M")
    plt.xlabel('Hands')
    plt.ylabel('Money')
    if betMethod == 'kelly':
        ax.set_yscale('log')

    # Vec subplots
    ax.plot(x_indices, vec_mean, label='Vec Mean', color='blue', linestyle='--')
    ax.plot(x_indices, vec_gmean, label='Vec Geo Mean', color='blue', linestyle='-')
    # HiLo subplots
    ax.plot(x_indices, hilo_mean, label='HiLo Mean', color='red', linestyle='--')
    ax.plot(x_indices, hilo_gmean, label='HiLo Geo Mean', color='red', linestyle='-')

    if betMethod == 'spread':
        plt.ticklabel_format(style='sci', axis='y', scilimits=(5,5))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'res/MoneyGraph_{betMethod}', dpi=500)

    # STD plots
    plt.figure(figsize=(15, 20)) 
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.85)
    fig, ax = plt.subplots()
    plt.xlabel('Hands')
    plt.ylabel('Money')
    plt.title(f"Mean with STD\nRunning {common.graphs_num_of_runs} games\nn_test = {common.n_test // common.MILLION}M")


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
    else:
        ax.fill_between(x_indices, vec_mean - vec_std, vec_mean + vec_std, alpha=0.2)
        ax.fill_between(x_indices, hilo_mean - hilo_std, hilo_mean + hilo_std, alpha=0.2)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'res/STDGraph_{betMethod}', dpi=500)

def MPBatchGamesWithArgs(betMethod, penetration=common.penetration, kellyFrac=common.kelly_fraction, spread=common.spread, queue=None, x = None):
    # print('start MultyProcess - Vec')
    # vecData,_ = multyProcessBatchGames(vec=True, betMethod=betMethod, penetration=penetration, kellyFrac=kellyFrac, spread=spread)
    # print('start MultyProcess - Hilo')
    # hiloData,_ = multyProcessBatchGames(vec=False, betMethod=betMethod,penetration=penetration, kellyFrac=kellyFrac, spread=spread)

    processes = []
    queue_2 = mp.Queue()
    for vec in [True, False]:
        p = mp.Process(target=multyProcessBatchGames, args=(vec, betMethod, penetration, kellyFrac, spread, queue_2))
        p.start()
        processes.append(p)

    results = dict()
    for p in processes:
        p.join()
        returned_values = queue_2.get()
        results[returned_values[0]] = returned_values[1:]

    vecData = results[True][0]
    hiloData = results[False][0]

    vec_mean = np.mean(vecData, axis=0)
    vec_gmean = gmean(vecData, axis=0)
    vec_std = np.std(vecData, axis=0)

    hilo_mean = np.mean(hiloData, axis=0)
    hilo_gmean = gmean(hiloData, axis=0)
    hilo_std = np.std(hiloData, axis=0)

    queue.put([vec_mean[-1], vec_gmean[-1], vec_std[-1], hilo_mean[-1], hilo_gmean[-1], hilo_std[-1], x])

def handleMPArgs(testedArg, x_indices, betMethod='kelly'):
    vecDatas_mean = dict()
    vecDatas_gmean = dict()
    vecDatas_std = dict()
    hiloDatas_mean = dict()
    hiloDatas_gmean = dict()
    hiloDatas_std = dict()

    results = []
    queue = mp.Queue()
    args = [betMethod, common.penetration, common.kelly_fraction, common.spread, queue]
    for x in x_indices:
        if testedArg == 'penetration':
            args[1] = x
        elif testedArg == 'kelly_fraction':
            args[2] = x
        elif testedArg == 'spread':
            args[3] = x
        else:
            raise Exception("WRONG testedArg")

        p = mp.Process(target=MPBatchGamesWithArgs, args=tuple(args + [x]))
        p.start()
        results.append(p)

    for p in results:
        p.join()
        returned_values = queue.get()
        x = returned_values[6]
        vecDatas_mean[x] = returned_values[0]
        vecDatas_gmean[x] = returned_values[1]
        vecDatas_std[x] = returned_values[2]
        hiloDatas_mean[x] = returned_values[3]
        hiloDatas_gmean[x] = returned_values[4]
        hiloDatas_std[x] = returned_values[5]

    return [vecDatas_mean, vecDatas_gmean, vecDatas_std, hiloDatas_mean, hiloDatas_gmean, hiloDatas_std]

def batchGamesAndPenetrate(betMethod='kelly'):
    a,b = common.penetration_sample_rate.split('/')
    pen_rate = float(a) / float(b)
    x_indices = np.arange(1,pen_rate,-pen_rate)

    results = handleMPArgs(testedArg='penetration', x_indices=x_indices, betMethod=betMethod)
    vecDatas_mean    = {k*6:v for k,v in results[0].items()}
    vecDatas_gmean   = {k*6:v for k,v in results[1].items()}
    vecDatas_std     = {k*6:v for k,v in results[2].items()}
    hiloDatas_mean   = {k*6:v for k,v in results[3].items()}
    hiloDatas_gmean  = {k*6:v for k,v in results[4].items()}
    hiloDatas_std    = {k*6:v for k,v in results[5].items()}

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

    x_indices = np.arange(0.2,0.9,common.kelly_frac_sample_rate)

    results = handleMPArgs(testedArg='kelly_fraction', x_indices=x_indices, betMethod='kelly')
    # vecDatas_mean = results[0]
    vecDatas_gmean = results[1]
    # vecDatas_std = results[2]
    # hiloDatas_mean = results[3]
    hiloDatas_gmean = results[4]
    # hiloDatas_std = results[5]

    fig, ax = plt.subplots()
    plt.title(f'Money - Kelly Fraction\nRunning {common.graphs_num_of_runs} games each')
    plt.xlabel(f'Kelly Fraction')
    # plt.xticks(np.arange(0, 2, 0.1))
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

    x_indices = np.arange(2,25,common.spread_sample_rate)

    results = handleMPArgs(testedArg='spread', x_indices=x_indices, betMethod='spread')
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
    # # Vec subplots
    # ax.plot(*zip(*sorted(vecDatas_mean.items())), label='Vec Mean', color='blue', linestyle='--')
    # # HiLo subplots
    # ax.plot(*zip(*sorted(hiloDatas_mean.items())), label='HiLo Mean', color='red', linestyle='--')
    #
    # ax.fill_between(x_indices, np.array(vecDatas_mean) - np.array(vecDatas_std), np.array(vecDatas_mean) + np.array(vecDatas_std), alpha=0.2)
    # ax.fill_between(x_indices, np.array(hiloDatas_mean) - np.array(hiloDatas_std), np.array(hiloDatas_mean) + np.array(hiloDatas_std), alpha=0.2)

    plt.legend()
    plt.grid(True)
    plt.savefig(f'res/MoneyPerSpread', dpi=500)

def run_create_vec():
    print("\nStarting run_create_vec")
    countAgent = CountAgent(needMatrixes=True, vec=True)
    for i in tqdm(range(common.n_test)):
        countAgent.runLoop(i)

    linear_reg(countAgent)


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
    p2 = mp.Process(target=batchGamesAndPenetrate, args=('kelly',))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


def main():
    common.winrateDictVec = loadFromDB('winrateDictVec')
    common.winrateDict = loadFromDB('winrateDict')
    setMaxMin(common.winrateDictVec, vec=True)
    setMaxMin(common.winrateDict, vec=True)

    # initializeDB()
    # run_create_vec()
    # multyprocessFinalTest()
    # multyprocessBatchAndFig()
    spreadGraph()
    multyprocessBatchGamesAndPenetrate()
    kellyFractionGraph()

    # for i in range(1):
    # x_indices = range(0,common.graphs_max_hands,1000)
    # (_, data, _) = singleGame(((0, False, "kelly", common.spread, common.penetration, 0.5)))
    # pprint(data[x_indices])

if __name__ == '__main__':
    main()
