import random
from builtins import print

cards_dict = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q",
              13: "K"}
cards_values = {1: 11, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10}

cards_count = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1, 12: -1, 13: -1}

class Shoe:
    def __init__(self, n=6):
        self.n = n
        self.cards = []
        self.rebuild()

    def draw_card(self):
        if not self.cards:
            self.rebuild()

        card = self.cards.pop(0)
        self.runningCount += cards_count[card]
        self.trueCount = self.runningCount # / self.n
        return card

    def rebuild(self):
        self.runningCount = 0
        self.trueCount = 0
        self.cards = [*range(1, 14)] * 4 * self.n
        random.shuffle(self.cards)


class Game:
    def __init__(self, money=100, dealerStandSoft17=True, splitAcesAndDone=False):
        self.money = money
        self.shoe = Shoe()
        self.dealerStandSoft17=dealerStandSoft17
        self.splitAcesAndDone = splitAcesAndDone

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

    # def manageBet(self, dealer_sum, player_sums, bet) -> int:
    #     retval = 0
    #     for i, _ in enumerate(self.playerCards):
    #         if dealer_sum == player_sums[i]:
    #             self.money += bet[i]
    #         elif dealer_sum > 21:
    #             self.money += 2 * bet[i]
    #             retval += 1 + self.isDouble
    #         elif dealer_sum < player_sums[i] <= 21:
    #             self.money += 2 * bet[i]
    #             retval += 1 + self.isDouble
    #         else:
    #             retval -= 1 + self.isDouble
    #
    #     return retval

    def rewardHandler(self, dealer_sum, player_sums):
        reward1 = self.__handRewadHandler(dealer_sum, player_sums[0])
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
        elif action == "P":
            self.isSplit = True
            self.playerCards.append([hand[1]])
            self.playerCards[0].pop()
            self.playerCards[0].append(self.shoe.draw_card())
            self.playerCards[1].append(self.shoe.draw_card())

        elif action == "X":
            reward = -0.5
            done = True

        sumHand, game_state = self.sum_hands(hand)

        if sumHand > 21 or action == "S" or action == "D" or (self.splitAcesAndDone and action == "P" and hand[0] == 1):
            reward, done = self.endGame()
            self.currHand = 1

        self.first_move = False

        return game_state, sumHand, reward, done

    def dealerMoves(self):
        # Dealer Moves
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
    
    def get_count(self):
        return self.shoe.trueCount

# def main():
#     game = Game()
#     win_diff=0
#     total=0
#     print("your current budget is ", game.money)
#     while (game.money > 0 and input("want to start new game? y/n\n") == "y"):
#         win_diff += game.run_game()
#         total+=1
#         print("win/lose = ", win_diff, " total games = ", total)
#         print("your current budget is ", game.money)
#
#
# if __name__ == '__main__':
#     main()
