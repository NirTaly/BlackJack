import learning
import simulator as sim
import basic_strategy as bs

def initCountDict():
    total_min_count_val = sim.COUNT_MIN_VAL_DECK * self.Game.shoe.n
    total_max_count_val = sim.COUNT_MAX_VAL_DECK * self.Game.shoe.n
    d = dict()

    for running_count in range(total_min_count_val, total_max_count_val+1):
        for decks in range(1,sim.Game.shoe.n):
            d[float(running_count)/decks] = 0
    return d

class CountAgent:
    def __init__(self):
        self.shoe.Game = sim.Game()
        self.Q_table = bs.initBasicStrategy()
        self.countDict = initCountDict()
    
    def handleBJ(self):
        reward, done = 0, False
        player_state, _ = self.Game.sum_hands(self.Game.playerCards[0])
        dealer_sum, _ = self.Game.sum_hands(self.Game.dealerCards)
        if player_state == 21 or dealer_sum == 21:
            reward = self.Game.rewardHandler(dealer_sum, [player_state])
            reward = 1.5 * reward if (reward > 0) else reward
            done = True
        return reward, done
    
    def getAction(self, game_state, player_state, dealer_state, explore=True):
        if self.Game.first_move:
            player_state = learning._playerStateFromGameState(self.Game, game_state, player_state)
            return max(self.Q_table[game_state][(player_state, dealer_state)],
                       key=self.Q_table[game_state][(player_state, dealer_state)].get)

        else:
            return max(dict(itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)),
                       key=dict(
                           itertools.islice(self.Q_table[game_state][(player_state, dealer_state)].items(), 2)).get)

    def run_loop(self):
        game_state, player_state = self.Game.reset_hands()
        count = self.Game.get_count()
        reward, done = self.handleBJ()
        for i, _ in enumerate(self.Game.playerCards):
            while not done:
                action = self.get_action(game_state, player_state, self.Game.dealer_state)

                next_game_state, next_player_state, reward, done = self.Game.step(action)

                game_state = next_game_state
                player_state = next_player_state

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

