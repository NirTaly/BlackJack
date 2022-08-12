import random
from builtins import print

cards_dict = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q",
              13: "K"}
cards_values = {1: 11, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10}


class Shoe:
    def __init__(self, n=8):
        self.n = n
        self.cards = []
        self.rebuild()

    def draw_card(self):
        if not self.cards:
            self.rebuild()

        return self.cards.pop(0)

    def rebuild(self):
        self.cards = [*range(1, 14)] * 4 * self.n
        random.shuffle(self.cards)


class Game:
    def __init__(self, money=100):
        self.money = money
        self.shoe = Shoe()
        self.reset_hands()

    def reset_hands(self):
        self.playerCards = [[]]
        self.dealerCards = []
        self.isSplit = False
        self.isDouble = False
        self.currHand = 0
        self.first_move = True

        # card draw
        for j in range(0, 2):
            self.playerCards[0].append(self.shoe.draw_card())
            self.dealerCards.append(self.shoe.draw_card())

        game_state = 0
        if 1 in self.playerCards[0]:
            game_state = 1
        player_state, _ = self.sum_hands(self.playerCards[0])
        dealer_state = self.dealerCards[0]
        return game_state, player_state, dealer_state

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
        soft = (hand.count(1) == 1)

        for i in range(hand.count(1)):
            if hand_sum > 21:
                hand_sum -= 10
        return hand_sum, soft

    def manageBet(self, dealer_sum, player_sums, bet) -> int:
        retval = 0
        for i, _ in enumerate(self.playerCards):
            if dealer_sum == player_sums[i]:
                self.money += bet[i]
            elif dealer_sum > 21:
                self.money += 2 * bet[i]
                retval += 1 + self.isDouble
            elif dealer_sum < player_sums[i] <= 21:
                self.money += 2 * bet[i]
                retval += 1 + self.isDouble
            else:
                retval -= 1 + self.isDouble

        return retval

    def rewardHandler(self, dealer_sum, player_sums) -> int:
        reward = 0
        for i, _ in enumerate(self.playerCards):
            if dealer_sum > 21:
                reward += 1 + self.isDouble
            elif dealer_sum < player_sums[i] <= 21:
                reward += 1 + self.isDouble
            else:
                reward -= 1 + self.isDouble

        return reward

    def run_game(self):
        self.reset_hands()

        # place bet
        bet = [float(input("please place a bet: "))]
        self.money -= bet[0]

        self.print_hands(False)

        # check BJ
        if sum(cards_values[i] for i in self.playerCards[0]) == 21:
            self.money += 2.5 * bet[0]
            return 1
        if sum(cards_values[i] for i in self.dealerCards) == 21:
            return -1

        # Player Moves
        first_move = True
        for hand in self.playerCards:
            while True:
                if first_move:
                    if hand[0] == hand[1]:
                        move = input("Select your move: Hit - H, Stand - S, Double - D, Split - P, Surrender - X\n")
                    else:
                        move = input("Select your move: Hit - H, Stand - S, Double - D, Surrender - X\n")
                else:
                    move = input("Select your move: Hit - H, Stand - S\n")
                move = move.upper()

                if move == "H":
                    hand.append(self.shoe.draw_card())
                    first_move = False
                    if self.isSplit and cards_dict[hand[0]] == "A":
                        break
                elif move == "S":
                    break
                elif move == "D" and first_move:
                    self.isDouble = True
                    self.money -= bet[0]
                    bet[0] *= 2

                    hand.append(self.shoe.draw_card())
                    break
                elif move == "P" and first_move and hand[0] == hand[1]:
                    self.isSplit = True
                    self.money -= bet[0]
                    bet += bet

                    self.playerCards.append([hand[1]])
                    self.playerCards[0].pop()

                    first_move = False
                elif move == "X":
                    self.money += 0.5 * bet[0]
                    return -1

                self.print_hands(False)

                if self.sum_hands(hand)[0] > 21:
                    if 1 == len(self.playerCards):
                        return -1
                    else:
                        break

        # Dealer Moves
        player_sums = []
        for i in range(len(self.playerCards)):
            player_sums.append(self.sum_hands(self.playerCards[i]))

        while True:
            self.print_hands()
            dealer_sum, _ = self.sum_hands(self.dealerCards)
            if 17 <= dealer_sum:
                break
            else:
                self.dealerCards.append(self.shoe.draw_card())

        return self.manageBet(dealer_sum, player_sums, bet)

    def step(self, action):
        hand = self.playerCards[self.currHand]
        reward = 0
        done = False

        if action == "H":
            hand.append(self.shoe.draw_card())
        elif action == "D":
            self.isDouble = True
            hand.append(self.shoe.draw_card())
        elif action == "P" and self.first_move and hand[0] == hand[1]:
            self.isSplit = True
            self.playerCards.append([hand[1]])
            self.playerCards[0].pop()
            self.playerCards[0].append(self.shoe.draw_card())
            self.playerCards[1].append(self.shoe.draw_card())

        elif (action == "P" and (not self.first_move or not hand[0] == hand[1])) or (action == "D" and not self.first_move):
            reward = -100000000
            done = True
            player_state, game_state = self.sum_hands(hand)
            dealer_state = self.dealerCards[0]
            return game_state, player_state, dealer_state, reward, done

        elif action == "X":
            reward = -0.5
            done = True

        sumHand, _ = self.sum_hands(hand)
        if sumHand > 21:
            if 1 == len(self.playerCards) and action == "D":
                reward = -2
                done = True
            elif 1 == len(self.playerCards):
                reward = -1
                done = True
            elif self.currHand == 0:
                self.currHand = 1
                hand = self.playerCards[self.currHand]
            else:
                reward, done = self.dealerMoves()

        elif action == "S" or action == "D" or (action == "P" and self.isSplit):
            reward, done = self.dealerMoves()
            self.currHand = 1

        self.first_move = False
        player_state, game_state = self.sum_hands(hand)
        dealer_state = self.dealerCards[0]

        return game_state, player_state, dealer_state, reward, done

    def dealerMoves(self):
        # Dealer Moves
        if self.currHand == 0 and self.isSplit:
            return 0, False
        player_sums = []
        for i in range(len(self.playerCards)):
            player_sum, _ = self.sum_hands(self.playerCards[i])
            player_sums.append(player_sum)

        while True:
            # self.print_hands()
            dealer_sum, _ = self.sum_hands(self.dealerCards)
            if 17 <= dealer_sum:
                break
            else:
                self.dealerCards.append(self.shoe.draw_card())

        return self.rewardHandler(dealer_sum, player_sums), True

# def main():
# 	game = Game()
# 	win_diff=0
# 	total=0
# 	print("your current budget is ", game.money)
# 	while (game.money > 0 and input("want to start new game? y/n\n") == "y"):
# 		win_diff += game.run_game()
# 		total+=1
# 		print("win/lose = ", win_diff, " total games = ", total)
# 		print("your current budget is ", game.money)


# if __name__ == '__main__':
# 	main()
