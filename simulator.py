import random

class Shoe:
	def __init__(self,n=8):
		self.n = n
		self.cards = []
		self.rebuild()
		
	def drawCard(self):
		return self.cards.pop(0)
  
	def rebuild(self):
		self.cards = [*range(1,14)] * 4*self.n
		random.shuffle(self.cards)
	
class Game:
	def __init__(self,money=100):
		self.money = money
		self.shoe = Shoe()
		