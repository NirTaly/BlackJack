import random

class Deck:
	def __init__(self):
		self.cards = []
		self.rebuild()
  
	def drawCard(self):
		return self.cards.pop(0)

	def rebuild(self):
		self.cards = [*range(1,14)] * 4
		random.shuffle(self.cards)

deck = Deck()
print (deck.cards)
print (deck.drawCard())
print (deck.drawCard())
print (deck.drawCard())
print (len(deck.cards))
deck.rebuild()
print (len(deck.cards))

class Shoe:
	pass

class Game:
    pass

