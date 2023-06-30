import common
import random
import numpy as np
from builtins import print

cards_dict = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q",
              13: "K"}
cards_values = {1: 11, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10}

cards_count = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1, 12: -1, 13: -1}

# cards_vec_count = {1: -9.633157680380947,
#  2: 5.865425788620057,
#  3: 6.819837964981865,
#  4: 8.671875332827337,
#  5: 10.889873899987279,
#  6: 6.555304867337135,
#  7: 3.4131097220163955,
#  8: -0.4359162635090514,
#  9: -2.9896812474366574,
#  10: -7.291308209106275}

#Bias = -0.0016153870383277535
cards_vec_count = {1: -10.005481541156769,
 2: 5.851210094988346,
 3: 6.978115998208523,
 4: 9.024945087730885,
 5: 11.382543481886387,
 6: 6.788263563066721,
 7: 3.3635925501585007,
 8: -0.764048658311367,
 9: -2.961400430649519,
 10: -7.4171628803014755}
#Sum of vector = -0.010911375284194946

##### semi-positive
# Bias = -0.005975118855549083
# cards_vec_count = {1: -10.25603805422867,
#                    2: 5.35561505383034,
#                    3: 6.061177211241191,
#                    4: 7.976949192967731,
#                    5: 9.880939496649942,
#                    6: 5.331453624069422,
#                    7: 3.5969764524585597,
#                    8: 0.051518498916456254,
#                    9: -2.1765897973080133,
#                    10: -6.406296322541003}

# Bias = 0.0031084460206329823
# cards_vec_count = {1: -10.18678117543459,
#  2: 4.614121280610561,
#  3: 5.492888391017914,
#  4: 7.501704152673483,
#  5: 10.03106776624918,
#  6: 5.173633806407452,
#  7: 3.567432751879096,
#  8: -0.11003308463841677,
#  9: -2.062728628516197,
#  10: -5.956897046416998}
# Sum of vector = 0.1968163884329428

########### NEW BASIC STRATEGY
# Bias = -0.0028472119010984898
# cards_vec_count =   {1: -2.42425873875618,
#                      2: 1.2439123820513487,
#                      3: 1.5906719490885735,
#                      4: 2.070463728159666,
#                      5: 2.5520019698888063,
#                      6: 1.3840324245393276,
#                      7: 0.7290207431651652,
#                      8: -0.11264040949754417,
#                      9: -0.6944899214431643,
#                      10: -1.6382205067202449}
# Sum of vector = -0.2141678996849805

class Shoe:
    def __init__(self, n=common.num_of_decks, penetration=common.penetration):
        self.n = n
        self.cards = []
        self.penetration = penetration
        self.rebuild()

    def draw_card(self):
        if not self.cards:
            self.rebuild()
        # elif len(self.cards) == round(common.DECK_SIZE * self.n * self.penetration):
        #     pass
        # elif len(self.cards) % (common.DECK_SIZE / 2) == 0:
        #     self.rem_decks -= 0.5
        self.rem_decks = self.currentRemaindedDecks()

        card = self.cards.pop(0)
        self.running_count += cards_count[card]

        if card >= 10:
            self.countVec[10] += 1
            self.running_count_vec += cards_vec_count[10]
        else:
            self.countVec[card] += 1
            self.running_count_vec += cards_vec_count[card]

        return card

    def rebuild(self):
        self.running_count = 0
        self.running_count_vec = 0
        self.countVec = np.zeros(11, dtype=np.uint8)
        self.countVec[0] = 1  # Bias always = 1
        self.rem_decks = self.n
        self.cards = [*range(1, 14)] * 4 * self.n

        random.shuffle(self.cards)

        # penetration
        num_of_cards_to_remove = round(common.DECK_SIZE * self.n * (1 - self.penetration))
        del self.cards[:num_of_cards_to_remove]
        self.rem_decks = self.currentRemaindedDecks()

    def getNormVec(self):
        divided = self.countVec / self.rem_decks
        divided[0] = 1
        # divided[10] = divided[10] / 4
        return divided

    def currentRemaindedDecks(self):
        return ((len(self.cards) // (common.DECK_SIZE / 2)) + 1) / 2

class Game:
    def __init__(self, decks=common.num_of_decks, penetration=common.penetration):
        self.first_move = True
        self.currHand = 0
        self.isDouble = [False]
        self.isSplit = False
        self.playerCards = [[]]
        self.dealerCards = []
        self.dealer_state = 0
        self.money = common.initial_money
        self.bet = 0
        self.total_bets = 0
        self.minBet = common.min_bet
        self.shoe = Shoe(decks, penetration)
        self.dealerStandSoft17 = common.dealerStandSoft17
        self.splitAcesAndDone = common.splitAcesAndDone

    def reset_hands(self):
        self.playerCards = [[]]
        self.dealerCards = []
        self.isSplit = False
        self.isDouble = [False]
        self.currHand = 0
        self.first_move = True

        # card draw
        for j in range(0, 2):
            self.playerCards[0].append(self.shoe.draw_card())
            self.dealerCards.append(self.shoe.draw_card())

        if self.playerCards[0][0] == self.playerCards[0][1]:
            game_state = 2
        elif self.playerCards[0].count(1) == 1:
            game_state = 1
        else:
            game_state = 0

        player_state, _ = self.sum_hands(self.playerCards[0])

        self.dealer_state = cards_values[self.dealerCards[0]]
        return game_state, player_state

    def sum_hands(self, hand):
        hand_sum = sum(cards_values[i] for i in hand)
        game_state = 0
        if hand[0] == hand[1] and len(hand) == 2:
            game_state = 2
        else:
            game_state = 1 if (hand.count(1) == 1 and hand_sum <= 21) else 0

        for i in range(hand.count(1)):
            if hand_sum > 21:
                hand_sum -= 10
        return hand_sum, game_state

    def manageBets(self, dealer_sum, player_sums):
        for i, _ in enumerate(player_sums):
            self.__manageBet(dealer_sum, i)

    def __manageBet(self, dealer_sum, player_hand_i):
        player_sum, _ = self.sum_hands(self.playerCards[player_hand_i])
        if dealer_sum == player_sum:
            self.money += self.bet + self.bet * self.isDouble[player_hand_i]
        elif dealer_sum < player_sum <= 21 or dealer_sum > 21:
            self.money += 2 * (self.bet + self.bet * self.isDouble[player_hand_i])
        else:
            pass

    def __handRewadHandler(self, dealer_sum, player_hand_i):
        reward = 0
        player_sum, _ = self.sum_hands(self.playerCards[player_hand_i])
        if player_sum > 21:
            reward -= 1 + self.isDouble[player_hand_i]
        elif dealer_sum > 21:
            reward += 1 + self.isDouble[player_hand_i]
        elif dealer_sum < player_sum:
            reward += 1 + self.isDouble[player_hand_i]
        elif player_sum == dealer_sum:
            reward = 0
        else:
            reward -= 1 + self.isDouble[player_hand_i]

        return reward

    def rewardHandler(self, dealer_sum, player_sums):
        rewards = 0
        for i, _ in enumerate(player_sums):
            rewards += self.__handRewadHandler(dealer_sum, i)
        self.manageBets(dealer_sum, player_sums)
        return rewards

    def step(self, action):
        hand = self.playerCards[self.currHand]
        reward = 0
        done = False

        if action == "H":
            hand.append(self.shoe.draw_card())
        elif action == "D":
            self.isDouble[self.currHand] = True
            hand.append(self.shoe.draw_card())
            self.money -= self.bet
            self.total_bets += self.bet
        elif action == "P":
            self.isSplit = True
            self.playerCards.append([hand[1]])
            self.playerCards[self.currHand].pop()
            self.playerCards[self.currHand].append(self.shoe.draw_card())
            self.playerCards[-1].append(self.shoe.draw_card())
            self.money -= self.bet
            self.total_bets += self.bet
            self.isDouble.append(False)
        elif action == "X":
            reward = -0.5
            self.money += 0.5 * self.bet
            done = True

        sumHand, game_state = self.sum_hands(hand)

        if sumHand > 21 or action == "S" or action == "D" or (self.splitAcesAndDone and action == "P" and hand[0] == 1):
            reward, done = self.endGame()
            self.currHand += 1
        self.first_move = False
        return game_state, sumHand, reward, done

    def dealerMoves(self):
        while True:
            dealer_sum, _ = self.sum_hands(self.dealerCards)
            if (17 < dealer_sum) or (17 == dealer_sum and self.dealerStandSoft17):
                break
            else:
                self.dealerCards.append(self.shoe.draw_card())
        return dealer_sum

    def endGame(self):
        if self.splitAcesAndDone:
            if self.currHand == 0 and self.isSplit and self.playerCards[0][0] != 1:
                return 0, True
        else:
            if self.currHand != len(self.playerCards) - 1 and self.isSplit:
                return 0, True

        player_sums = []
        not_all_hands_burned = True
        for i in range(len(self.playerCards)):
            player_sum, _ = self.sum_hands(self.playerCards[i])
            player_sums.append(player_sum)
            if player_sum <= 21:
                not_all_hands_burned = False

        # Dealer Moves
        dealer_sum, _ = self.sum_hands(self.dealerCards)
        if not not_all_hands_burned:
            dealer_sum = self.dealerMoves()

        return self.rewardHandler(dealer_sum, player_sums), True

    def get_count(self, vec=False):
        if vec:
            return float(self.shoe.running_count_vec) / self.shoe.rem_decks
        else:
            return float(self.shoe.running_count) / self.shoe.rem_decks

    def place_bet(self, bet):
        self.bet = bet
        self.total_bets += bet
        self.money -= bet

    def handleBJ(self):
        reward, done = 0, False
        player_state, _ = self.sum_hands(self.playerCards[0])
        dealer_sum, _ = self.sum_hands(self.dealerCards)
        if player_state == 21 or dealer_sum == 21:
            reward = self.rewardHandler(dealer_sum, [player_state])
            if reward > 0:
                self.money += (5 / 2) * self.bet
                reward *= 1.5
            done = True
        return reward, done

    def isAfterSplit(self):
        return self.isSplit and len(self.playerCards[self.currHand]) == 2

    def print_hands(self, showDealerCard=True):
        dealer_str = "Dealer's Cards: "
        player_str = "Your Cards:     "
        if not showDealerCard:
            dealer_str += "* "
        else:
            dealer_str += cards_dict[self.dealerCards[0]]
            dealer_str += " "
        for card in self.dealerCards[1:]:
            dealer_str += cards_dict[card]
            dealer_str += " "

        if showDealerCard:
            print(dealer_str + "\t\t\t\tsum = " + str(self.sum_hands(self.dealerCards)))
        else:
            print(dealer_str + "\t\t\t\tsum = " + str(cards_values[self.dealerCards[1]]))

        for i, hand in enumerate(self.playerCards):
            if len(self.playerCards) > 1:
                player_str = "Hand " + str(i) + ": "
            for card in hand:
                player_str += cards_dict[card]
                player_str += " "
            player_str += "\t\t\t\tsum = " + str(self.sum_hands(hand))
            print(player_str)
            print("")


def main():
    game = Game()
    print("your current budget is ", game.money)
    while game.money >= game.minBet:
        game.reset_hands()
        # game.place_bet(int(input("place bet - min: " + str(game.minBet) + "\tmax: " + str(game.money) + ":\t")))
        game.place_bet(5)
        game.print_hands(False)
        _, done = game.handleBJ()
        for _ in game.playerCards:
            while not done:
                if len(game.playerCards) == 3:
                    _, _, _, done = game.step(str(input('Action: ')))
                else:
                    if (game.playerCards[0][0] != game.playerCards[0][1]):
                        _, _, _, done = game.step('S')
                    else:
                        _, _, _, done = game.step('P')
                game.print_hands()
            if game.isSplit:
                if game.playerCards[0][0] == 1 and game.splitAcesAndDone:
                    break
                done = False

        print("your current budget is ", game.money)


if __name__ == '__main__':
    main()
