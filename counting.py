import learning
import simulator as sim
import basic_strategy as bs
import common

def initCountDict(game):
    total_min_count_val = common.COUNT_MIN_VAL_DECK * game.shoe.n
    total_max_count_val = common.COUNT_MAX_VAL_DECK * game.shoe.n
    d = dict()

    for running_count in range(total_min_count_val, total_max_count_val+1):
        for decks in range(1,game.shoe.n):
            d[float(running_count)/decks] = 0
    return d

class CountAgent:
    def __init__(self):
        self.game = sim.Game()
        self.Q_table = bs.initBasicStrategy()
        self.countDict = initCountDict()
    
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
            player_state = learning._playerStateFromGameState(self.game, game_state, player_state)
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

            if self.Game.isSplit:
                if self.Game.playerCards[0][0] == 1 and self.Game.splitAcesAndDone:
                    break
                player_state, game_state = self.Game.sum_hands(self.Game.playerCards[self.Game.currHand])
                done = False
            else:
                rewards = reward

        if self.Game.isSplit:
            rewards = sum(reward)
            wins = sum(1 for w in reward if w > 0)
        else:
            wins = 1 if (reward > 0) else 0

        self.countDict[count] += rewards
        return rewards, wins

def batchGames(countAgent):
    self.game.shoe.rebuild()
    while self.game.money > self.game.minBet:
        self.runLoop()

def finalTest():
    countAgent = CountAgent()
    for _ in tqdm(range(1, common.n_test+1)):
        reward, wins = self.run_loop(False)

def main():
    finalTest()

if __name__ == '__main__':
    main()
