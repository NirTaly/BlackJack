import random
from builtins import print

cards_dict = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K"}
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
		self.player_cards = []
		self.dealer_cards = []

	def reset_hands(self):
		self.player_cards = []
		self.dealer_cards = []

	def print_hands(self, showDealerCard = True):
		dealer_str = "Dealer's Cards: "
		player_str = "Your Cards:     "
		if not showDealerCard:
			dealer_str += "* "
		else:
			dealer_str += cards_dict[self.dealer_cards[0]]
			dealer_str += " "
		for card in self.dealer_cards[1:]:
			dealer_str += cards_dict[card]
			dealer_str += " "
		for card in self.player_cards:
			player_str += cards_dict[card]
			player_str += " "

		if showDealerCard:
			print(dealer_str + "\t\t\t\tsum = " + str(self.sum_hands(self.dealer_cards)))
		else:
			print(dealer_str + "\t\t\t\tsum = " + str(cards_values[self.dealer_cards[1]]))

		print(player_str + "\t\t\t\tsum = " + str(self.sum_hands(self.player_cards)))

	def sum_hands(self, hand):
		sum = 0
		for card in hand:
			sum += cards_values[card]
		#Ace is soft!
		if 1 in hand and sum > 21:
			sum -= 10
		return sum

	def run_game(self):
		self.reset_hands()
		for j in range(0, 2):
			self.player_cards.append(self.shoe.draw_card())
			self.dealer_cards.append(self.shoe.draw_card())

		self.print_hands(False)

		#Player Moves
		first_move = True
		move = ""
		while True:
			if first_move:
				move = input("Select your move: Hit - H, Stand - S, Double down - D, Split - P\n")
			else:
				move = input("Select your move: Hit - H, Stand - S\n")

			if move.upper() == "H":
				self.player_cards.append(self.shoe.draw_card())
				first_move = False
			if move.upper() == "S":
				break
			if first_move and self.player_cards[0] == self.player_cards[1] and (move == "D" or move == "P"):
				first_hand = [self.player_cards[0]]
				second_hand = [self.player_cards[1]]
				self.player_cards = first_hand + second_hand
				if move.upper() == "D":
					pass
				if move.upper() == "P":
					pass
				first_move = False

			self.print_hands(False)

			if self.sum_hands(self.player_cards) > 21:
				return -1



		#Dealer Moves
		player_sum = self.sum_hands(self.player_cards)
		while True:
			self.print_hands()
			dealer_sum = self.sum_hands(self.dealer_cards)
			if dealer_sum > 21 or 17 < dealer_sum < player_sum:
				return 1
			elif dealer_sum >= player_sum:
				return -1
			else:
				self.dealer_cards.append(self.shoe.draw_card())

game = Game()
win_diff=0
total=0
while (input("want to start new game? y/n\n") == "y"):
	win_diff += game.run_game()
	total+=1
	print("win/lose = ", win_diff, " total games = ", total)