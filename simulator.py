import common
import random
import numpy as np
from builtins import print

cards_dict = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q",
              13: "K"}
cards_values = {1: 11, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10}

cards_count = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1, 12: -1, 13: -1}


#Bias = 0.006928951467022775
cards_vec_count = {1: -10.559050335551536,
 2: 6.626549835944278,
 3: 7.643277584657614,
 4: 9.394683388516347,
 5: 12.613992633565802,
 6: 7.642751642233357,
 7: 3.725994331219095,
 8: -1.0408825072432364,
 9: -3.4816847223829885,
 10: -8.141558921741513}

#Sum of vector = 0.00040542901362774705

#cards_vec_count = {1: -12.641185102947892,
# 2: 10.029463632237185,
# 3: 5.812862245054516,
# 4: 7.041287327375578,
# 5: 15.278964481036708,
# 6: 6.148741500515471,
# 7: 2.84588579861269,
# 8: 1.6633612469682408,
# 9: -7.0623839111709295,
# 10: -7.840853569530667}

# cards_vec_count = {1: -11.042615985359793,
#  2: 6.125676361677346,
#  3: 8.360637600022915,
#  4: 9.582956807877073,
#  5: 12.74690239179271,
#  6: 6.662251733078631,
#  7: 4.268685514389877,
#  8: -0.679753439637321,
#  9: -3.3050215937721337,
#  10: -8.372830792509397,
#  11: -8.105644040887418,
#  12: -8.085158056021344,
#  13: -8.141166708328461}


class Shoe:
    def __init__(self, n=common.num_of_decks):
        self.n = n
        self.cards = []
        self.rebuild()

    def draw_card(self):
        if not self.cards:
            self.rebuild()
        elif len(self.cards) == common.DECK_SIZE * common.num_of_decks:
            pass
        elif len(self.cards) % common.DECK_SIZE == 0:
            self.rem_decks -= 1

        card = self.cards.pop(0)
        self.running_count += cards_count[card]
        
        if card >= 10:
            self.countVec[10] += 1
            self.running_count_vec += cards_vec_count[10]
        else:
            self.countVec[card] += 1
            self.running_count_vec += cards_vec_count[card]

        # self.running_count_vec += cards_vec_count[card]
        # self.countVec[card] += 1
        return card

    def rebuild(self):
        self.running_count = 0
        self.running_count_vec = 0
        self.countVec = np.zeros(11,dtype=np.uint8)
        # self.countVec = np.zeros(14,dtype=np.uint8)
        self.countVec[0] = 1 # Bias always = 1
        self.rem_decks = self.n
        self.cards = [*range(1, 14)] * 4 * self.n
        random.shuffle(self.cards)

    def getNormVec(self):
        divided = self.countVec / self.rem_decks
        divided[0] = 1
        divided[10] = divided[10] / 4
        # return divided.round()
        return divided
    



class Game:
    def __init__(self):
        self.money = common.initial_money
        self.bet = 0
        self.total_bets = 0
        self.minBet = common.min_bet
        self.shoe = Shoe()
        self.dealerStandSoft17 = common.dealerStandSoft17
        self.splitAcesAndDone = common.splitAcesAndDone

    def __handRewadHandler(self, dealer_sum, player_sum):
        reward = 0
        if player_sum > 21:
            reward -= 1 + self.isDouble
        elif dealer_sum > 21:
            reward += 1 + self.isDouble
        elif dealer_sum < player_sum:
            reward += 1 + self.isDouble
        elif player_sum == dealer_sum:
            reward = 0
        else:
            reward -= 1 + self.isDouble

        return reward

    def reset_hands(self, onlyPairs=False):
        self.playerCards = [[]]
        self.dealerCards = []
        self.isSplit = False
        self.isDouble = False
        self.currHand = 0
        self.first_move = True
        # self.bet = 0

        # card draw
        if not onlyPairs:
            for j in range(0, 2):
                self.playerCards[0].append(self.shoe.draw_card())
                self.dealerCards.append(self.shoe.draw_card())

            if (self.playerCards[0][0] == self.playerCards[0][1]):
                game_state = 2
            elif (self.playerCards[0].count(1) == 1):
                game_state = 1
            else:
                game_state = 0

            player_state, _ = self.sum_hands(self.playerCards[0])
        else:
            card = random.randint(1,13)
            self.playerCards[0] = [card, card]
            game_state = 2
            self.dealerCards = [self.shoe.draw_card(), self.shoe.draw_card()]
            player_state = 2 * cards_values[self.playerCards[0][0]]

        self.dealer_state = cards_values[self.dealerCards[0]]
        return game_state, player_state

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

    def sum_hands(self, hand):
        hand_sum = sum(cards_values[i] for i in hand)
        game_state = 1 if (hand.count(1) == 1 and hand_sum <= 21) else 0

        for i in range(hand.count(1)):
            if hand_sum > 21:
                hand_sum -= 10
        return hand_sum, game_state

    def manageBets(self, dealer_sum, player_sums):
        for player_sum in player_sums:
            self.__manageBet(dealer_sum, player_sum)

    def __manageBet(self, dealer_sum, player_sum):
        if dealer_sum == player_sum:
            self.money += self.bet
        elif dealer_sum < player_sum <= 21 or dealer_sum > 21:
            self.money += 2 * self.bet
        else:
            pass

    def rewardHandler(self, dealer_sum, player_sums):
        reward1 = self.__handRewadHandler(dealer_sum, player_sums[0])
        self.manageBets(dealer_sum, player_sums)
        if len(player_sums) == 2:
            reward2 = self.__handRewadHandler(dealer_sum, player_sums[1])
            reward_tuple = (reward1, reward2)
            return reward_tuple

        return reward1

    def step(self, action):
        hand = self.playerCards[self.currHand]
        reward = 0
        done = False

        if action == "H":
            hand.append(self.shoe.draw_card())
        elif action == "D":
            self.isDouble = True
            hand.append(self.shoe.draw_card())
            self.money -= self.bet
            self.bet *= 2
            self.total_bets += self.bet
            # self.place_bet(self.bet)
        elif action == "P":
            self.isSplit = True
            self.playerCards.append([hand[1]])
            self.playerCards[0].pop()
            self.playerCards[0].append(self.shoe.draw_card())
            self.playerCards[1].append(self.shoe.draw_card())
            self.money -= self.bet
            self.total_bets += self.bet
        elif action == "X":
            reward = -0.5
            self.money += 0.5 * self.bet
            done = True

        sumHand, game_state = self.sum_hands(hand)

        if sumHand > 21 or action == "S" or action == "D" or (self.splitAcesAndDone and action == "P" and hand[0] == 1):
            reward, done = self.endGame()
            self.currHand = 1

        self.first_move = False

        return game_state, sumHand, reward, done

    def dealerMoves(self):
        while True:
            dealer_sum, _ = self.sum_hands(self.dealerCards)
            if (17 < dealer_sum) or  (17 == dealer_sum and self.dealerStandSoft17):
                break
            else:
                self.dealerCards.append(self.shoe.draw_card())
        return dealer_sum

    def endGame(self):
        if self.splitAcesAndDone:
            if self.currHand == 0 and self.isSplit and self.playerCards[0][0] != 1:
                return 0, True
        else:
            if self.currHand == 0 and self.isSplit:
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
    
    def get_count(self,vec = False):
        if vec:
            return float(self.shoe.running_count_vec) / self.shoe.rem_decks
        else:
            return float(self.shoe.running_count) / self.shoe.rem_decks
    
    def place_bet(self, bet):
        #self.bet = bet
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
                self.game.money += (5/2) * self.game.bet
                reward *= 1.5
            done = True
        return reward, done


def main():
    game = Game()
    done=0
    print("your current budget is ", game.money)
    while (game.money >= game.minBet):
        game.reset_hands()
        game.place_bet(int(input("place bet - min: " + str(game.minBet) + "\tmax: " + str(game.money) + ":\t")))
        game.print_hands(False)
        _ , done = game.handleBJ()
        for _ in game.playerCards:
            while not done:
                _, _, _, done = game.step(str((input("action:\t"))).upper())
                game.print_hands()
            if game.isSplit:
                if game.playerCards[0][0] == 1 and game.splitAcesAndDone:
                    break
                done = False

        print("your current budget is ", game.money)
        done = False


if __name__ == '__main__':
    main()
