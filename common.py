# Constants
DECK_SIZE = 4*13              #13 cards, each have 4 shapes
COUNT_MAX_VAL_DECK = (1)*5*4  #value of  1, 5 cards ([2..6]), each have 4 shapes
COUNT_MIN_VAL_DECK = (-1)*5*4 #value of -1, 5 cards ([2..6]), each have 4 shapes
num_of_decks = 6
initial_money = 100
min_bet = initial_money // 20
MILLION = 1000000

n_learning = 30 * MILLION
n_train = 10 * MILLION
n_test = int(200 * MILLION)

# Rules
dealerStandSoft17 = True
splitAcesAndDone = False
