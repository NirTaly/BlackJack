import learning

################################# tables for BS2 #################################
hard_table = [
        # 2    3    4    5    6    7    8    9    10   A
        ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],  # 4
        ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],  # 5
        ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],  # 6
        ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],  # 7
        ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],  # 8
        ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # 9
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H'],  # 10
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H'],  # 11
        ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # 12
        ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # 13
        ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # 14
        ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'X', 'H'],  # 15
        ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'X', 'X', 'X'],  # 16
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 17
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 18
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 19
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 20
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']   # 21
    ]
soft_table = [
        # 2    3    4    5    6    7    8    9    10   A
        ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # 13
        ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # 14
        ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # 15
        ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # 16
        ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # 17
        ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'H', 'H', 'H'],  # 18
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 19
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 20
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']   # 21
    ]
split_table = [
        # 2    3    4    5    6    7    8    9    10   A
        ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H'],  # 2
        ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H'],  # 3
        ['H', 'H', 'H', 'P', 'P', 'H', 'H', 'H', 'H', 'H'],  # 4
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H'],  # 5
        ['P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H', 'H'],  # 6
        ['P', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H'],  # 7
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],  # 8
        ['P', 'P', 'P', 'P', 'P', 'S', 'P', 'P', 'S', 'S'],  # 9
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # 10
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']   # A
    ]
################################# tables for BS2 #################################

# old bs
def BS1():
    Q_table_hard = learning.CreateSimpleQTable((4, 22, 1))
    Q_table_soft = learning.CreateSimpleQTable((13, 22, 1))
    Q_table_split = learning.CreateSimpleQTable((2, 12, 1))

    # -- HARD --

    # 4-8 against 2-10 *HIT*
    for p in range(4, 12):
        for d in range(2, 12):
            Q_table_hard[(p, d)] = 'H'

    # 9 against 3-6 *DOUBLE*
    for d in range(3, 7):
        Q_table_hard[(9, d)] = 'D'

    # 10,11 against 2-9 *DOUBLE*
    for p in range(10, 12):
        for d in range(2, 10):
            Q_table_hard[(p, d)] = 'D'

    # 11 against 10 *DOUBLE*
    Q_table_hard[(11, 10)] = 'D'

    # 10,11 against 10,A *HIT*
    Q_table_hard[(10, 10)] = 'H'
    Q_table_hard[(10, 11)] = 'H'
    Q_table_hard[(11, 11)] = 'H'

    # 12 against 2,3,7,8,9,10 *HIT*
    for d in [2, 3, 7, 8, 9, 10, 11]:
        Q_table_hard[(12, d)] = 'H'

    # 12 against 4-6 *STAND*
    for d in range(4, 7):
        Q_table_hard[(12, d)] = 'S'

    # 13-16 against 2-6 *STAND*
    for p in range(13, 17):
        for d in range(2, 7):
            Q_table_hard[(p, d)] = 'S'

    # 13 against 7-10 *HIT*
    for p in range(12, 17):
        for d in range(7, 12):
            Q_table_hard[(p, d)] = 'H'

    # 14,15 against 10,A *SURRENDER*
    for d in range(9, 12):
        Q_table_hard[(16, d)] = 'X'
    Q_table_hard[(15, 10)] = 'X'

    # 17 against 2-10 *STAND*
    for d in range(2, 12):
        Q_table_hard[(17, d)] = 'S'

    # 18+ against ALL *STAND*
    for p in range(18, 22):
        for d in range(2, 12):
            Q_table_hard[(p, d)] = 'S'

    # -- SOFT --
    # 13,14 against 2-4,7-11 *HIT*
    for p in range(13, 18):
        for d in range(2, 12):
            Q_table_soft[(p, d)] = 'H'

    # 13,14 against 5,6 *DOUBLE*
    for p in range(13, 15):
        for d in range(5, 7):
            Q_table_soft[(p, d)] = 'D'

    # 15,16 against 4,5,6 *DOUBLE*
    for p in range(15, 17):
        for d in range(4, 7):
            Q_table_soft[(p, d)] = 'D'

    # 17,18 against 3-6 *DOUBLE*
    for p in range(17, 19):
        for d in range(3, 7):
            Q_table_soft[(p, d)] = 'D'

    # 18 against 2,7,8 *STAND*
    for d in range(2, 9):
        Q_table_soft[(18, d)] = 'S'

    # 18 against 9-11 *HIT*
    for d in range(9, 12):
        Q_table_soft[(18, d)] = 'H'

    # 19-21 against ALL *STAND*
    for p in range(19, 22):
        for d in range(2, 12):
            Q_table_soft[(p, d)] = 'S'

    # -- SPLIT --

    # 2,3 against 2,3,8,9,10,A *HIT*
    for p in range(2, 4):
        for d in [2, 3, 8, 9, 10, 11]:
            Q_table_split[(p, d)] = 'H'

    # 2,3 against 4-7 *SPLIT*
    for p in range(2, 4):
        for d in range(4, 8):
            Q_table_split[(p, d)] = 'P'

    # correct 3 against A *SURRENDER*
    Q_table_split[(3, 11)] = 'X'

    # 4 against ALL *HIT*
    for d in range(2, 12):
        Q_table_split[(4, d)] = 'H'

    # 5 against 2-9 *DOUBLE*
    for d in range(2, 10):
        Q_table_split[(5, d)] = 'D'

    # 5 against 10,11 *HIT*
    for d in range(2, 12):
        Q_table_split[(5, d)] = 'H'

    # 6 against 2,7,8,9,10 *HIT*
    for d in [2, 7, 8, 9, 10, 11]:
        Q_table_split[(6, d)] = 'H'

    # 6 against 3,4,5,6 *SPLIT*
    for d in [3, 4, 5, 6]:
        Q_table_split[(6, d)] = 'P'

    # 7 against 8-A *HIT*
    for d in [8, 9, 10, 11]:
        Q_table_split[(7, d)] = 'H'

    # 7 against 2,3,4,5,6,7 *SPLIT*
    for d in [2, 3, 4, 5, 6, 7]:
        Q_table_split[(7, d)] = 'P'

    # 8 against 2,3,4,5,6,7,8,9 *SPLIT*
    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        Q_table_split[(8, d)] = 'P'

    # 9 against 2,3,4,5,6,8,9 *SPLIT*
    for d in [2, 3, 4, 5, 6, 8, 9]:
        Q_table_split[(9, d)] = 'P'

    # 9 against A *STAND*
    for d in [7, 10, 11]:
        Q_table_split[(9, d)] = 'S'

    # 10 against ALL *STAND*
    for d in range(2, 12):
        Q_table_split[(10, d)] = 'S'

    # A against 2-10 *SPLIT*
    for d in range(2, 12):
        Q_table_split[(11, d)] = 'P'

    return [Q_table_hard, Q_table_soft, Q_table_split]

def BS2():
    Q_table_hard = learning.CreateSimpleQTable((4, 22, 1))
    Q_table_soft = learning.CreateSimpleQTable((13, 22, 1))
    Q_table_split = learning.CreateSimpleQTable((2, 12, 1))


    for player_state in range(4,22):
        for dealer_sate in range(2, 12):
            Q_table_hard[(player_state, dealer_sate)] = hard_table[player_state-4][dealer_sate-2]
    for player_state in range(13,22):
        for dealer_sate in range(2, 12):
            Q_table_soft[(player_state, dealer_sate)] = soft_table[player_state-13][dealer_sate-2]
    for player_state in range(2,12):
        for dealer_sate in range(2, 12):
            Q_table_split[(player_state, dealer_sate)] = split_table[player_state-2][dealer_sate-2]

    return [Q_table_hard, Q_table_soft, Q_table_split]


def initBasicStrategy(bs):
    if bs == 'reg':
        return BS1()
    elif bs == 'das_sas':
        return BS2()
    else:
        return None
