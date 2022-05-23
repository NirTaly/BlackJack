import random
from builtins import print

cards_dict={1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K"}
cards_values={1: 11, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10}


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
		self.playerCards = [[]]
		self.dealerCards = []
		self.isSplit = False

	def reset_hands(self):
		self.playerCards = [[]]
		self.dealerCards = []
		self.isSplit = False

	def print_hands(self, showDealerCard = True):
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

		#Ace is soft!
		for i in range(hand.count(1)):
			if hand_sum > 21:
				hand_sum -= 10
		return hand_sum

	def manageBet(self, dealer_sum, player_sum, move, bet):
		if self.isSplit:
			#TODO
			pass
		else:
			if dealer_sum == player_sum:
				self.money += bet[0]
				return 0
			elif dealer_sum > 21:
				self.money += 2*bet[0]
			elif player_sum > dealer_sum and player_sum <= 21:
				self.money += 2*bet[0]



	def run_game(self):
		self.reset_hands()

		#place bet
		bet = [float(input("please place a bet: "))]
		self.money -= bet[0]

		#card draw
		for j in range(0, 2):
			self.playerCards[0].append(self.shoe.draw_card())
			self.dealerCards.append(self.shoe.draw_card())

		self.print_hands(False)

		#check BJ
		if sum(cards_values[i] for i in self.playerCards[0]) == 21:
			self.money += 2.5*bet[0]
			return 1

		#Player Moves
		first_move = True
		move = ""
		for hand in self.playerCards:
			while True:
				if first_move:
					if hand[0] == hand[1]:
						move = input("Select your move: Hit - H, Stand - S, Double - D, Split - P, Surrender - X\n")
					else:
						move = input("Select your move: Hit - H, Stand - S, Double - D, Surrender - X\n")
				else:
					move = input("Select your move: Hit - H, Stand - S\n")
				move= move.upper()

				if move == "H":
					hand.append(self.shoe.draw_card())
					first_move = False
					if self.isSplit and cards_dict[hand[0]] == "A":
						break
				elif move == "S":
					break
				elif move == "D" and first_move:
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
					self.money += 0.5*bet[0]
					return -1

				self.print_hands(False)

				if self.sum_hands(hand) > 21:
					return -1

		#Dealer Moves
		player_sum = self.sum_hands(self.playerCards[0]) 		#TODO handle for split (2 hands)
		player_win = True
		while True:
			self.print_hands()
			dealer_sum = self.sum_hands(self.dealerCards)
			if 17 <= dealer_sum:
				if dealer_sum > 21 or dealer_sum <= player_sum:
					break
				else:
					player_win = False
					break
			else:
				self.dealerCards.append(self.shoe.draw_card())

		self.manageBet(dealer_sum, player_sum, move, bet)
		return 2*int(player_win)-1	#return 1 if player won, -1 if lose

def main():
	game = Game()
	win_diff=0
	total=0
	print("your current budget is ", game.money)
	while (game.money > 0 and input("want to start new game? y/n\n") == "y"):
		win_diff += game.run_game()
		total+=1
		print("win/lose = ", win_diff, " total games = ", total)
		print("your current budget is ", game.money)


if __name__ == '__main__':
	main()