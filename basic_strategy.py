import learning

def initBasicStrategy():
    Q_table_hard = learning.CreateQTable((4, 22, 1), ['S', 'H', 'X', 'D'])
    Q_table_soft = learning.CreateQTable((13, 22, 1), ['S', 'H', 'X', 'D'])
    Q_table_split = learning.CreateQTable((2, 12, 1), ['S', 'H', 'X', 'D', 'P'])

    # -- HARD --

    # 4-8 against 2-10 *HIT*
    for p in range(4, 12):
        for d in range(2, 12):
            Q_table_hard[(p, d)]['H'] = 1

    # 9 against 3-6 *DOUBLE*
    for d in range(3, 7):
        Q_table_hard[(9, d)]['D'] = 2

    # 10,11 against 2-9 *DOUBLE*
    for p in range(10, 12):
        for d in range(2, 10):
            Q_table_hard[(p, d)]['D'] = 2

    # 11 against 10 *DOUBLE*
    Q_table_hard[(11, 10)]['D'] = 2

    # 10,11 against 10,A *HIT*
    Q_table_hard[(10, 10)]['H'] = 1
    Q_table_hard[(10, 11)]['H'] = 1
    Q_table_hard[(11, 11)]['H'] = 1

    # 12 against 2,3,7,8,9,10 *HIT*
    for d in [2, 3, 7, 8, 9, 10, 11]:
        Q_table_hard[(12, d)]['H'] = 1

    # 12 against 4-6 *STAND*
    for d in range(4, 7):
        Q_table_hard[(12, d)]['S'] = 1

    # 13-16 against 2-6 *STAND*
    for p in range(13, 17):
        for d in range(2, 7):
            Q_table_hard[(p, d)]['S'] = 1

    # 13 against 7-10 *HIT*
    for p in range(12, 17):
        for d in range(7, 12):
            Q_table_hard[(p, d)]['H'] = 1

    # 14,15 against 10,A *SURRENDER*
    for d in range(9, 12):
        Q_table_hard[(16, d)]['X'] = 2
    Q_table_hard[(15, 10)]['X'] = 2

    # 17 against 2-10 *STAND*
    for d in range(2, 12):
        Q_table_hard[(17, d)]['S'] = 1

    # 18+ against ALL *STAND*
    for p in range(18, 22):
        for d in range(2, 12):
            Q_table_hard[(p, d)]['S'] = 1

    # -- SOFT --
    # 13,14 against 2-4,7-11 *HIT*
    for p in range(13, 18):
        for d in range(2, 12):
            Q_table_soft[(p, d)]['H'] = 1

    # 13,14 against 5,6 *DOUBLE*
    for p in range(13, 15):
        for d in range(5, 7):
            Q_table_soft[(p, d)]['D'] = 2

    # 15,16 against 4,5,6 *DOUBLE*
    for p in range(15, 17):
        for d in range(4, 7):
            Q_table_soft[(p, d)]['D'] = 2

    # 17,18 against 3-6 *DOUBLE*
    for p in range(17, 19):
        for d in range(3, 7):
            Q_table_soft[(p, d)]['D'] = 2

    # 18 against 2,7,8 *STAND*
    for d in range(2, 9):
        Q_table_soft[(18, d)]['S'] = 1

    # 18 against 9-11 *HIT*
    for d in range(9, 12):
        Q_table_soft[(18, d)]['H'] = 1

    # 19-21 against ALL *STAND*
    for p in range(19, 22):
        for d in range(2, 12):
            Q_table_soft[(p, d)]['S'] = 1

    # -- SPLIT --

    # 2,3 against 2,3,8,9,10,A *HIT*
    for p in range(2, 4):
        for d in [2, 3, 8, 9, 10, 11]:
            Q_table_split[(p, d)]['H'] = 1

    # 2,3 against 4-7 *SPLIT*
    for p in range(2, 4):
        for d in range(4, 8):
            Q_table_split[(p, d)]['P'] = 1

    # correct 3 against A *SURRENDER*
    Q_table_split[(3, 11)]['X'] = 1

    # 4 against ALL *HIT*
    for d in range(2, 12):
        Q_table_split[(4, d)]['H'] = 1

    # 5 against 2-9 *DOUBLE*
    for d in range(2, 10):
        Q_table_split[(5, d)]['D'] = 2

    # 5 against 10,11 *HIT*
    for d in range(2, 12):
        Q_table_split[(5, d)]['H'] = 1

    # 6 against 2,7,8,9,10 *HIT*
    for d in [2, 7, 8, 9, 10, 11]:
        Q_table_split[(6, d)]['H'] = 1

    # 6 against 3,4,5,6 *SPLIT*
    for d in [3, 4, 5, 6]:
        Q_table_split[(6, d)]['P'] = 1

    # 7 against 8-A *HIT*
    for d in [8, 9, 10, 11]:
        Q_table_split[(7, d)]['H'] = 1

    # 7 against 2,3,4,5,6,7 *SPLIT*
    for d in [2, 3, 4, 5, 6, 7]:
        Q_table_split[(7, d)]['P'] = 1

    # 8 against 2,3,4,5,6,7,8,9 *SPLIT*
    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        Q_table_split[(8, d)]['P'] = 1

    # 9 against 2,3,4,5,6,8,9 *SPLIT*
    for d in [2, 3, 4, 5, 6, 8, 9]:
        Q_table_split[(9, d)]['P'] = 1

    # 9 against A *STAND*
    for d in [7, 10, 11]:
        Q_table_split[(9, d)]['S'] = 1

    # 10 against ALL *STAND*
    for d in range(2, 12):
        Q_table_split[(10, d)]['S'] = 1

    # A against 2-10 *SPLIT*
    for d in range(2, 12):
        Q_table_split[(11, d)]['P'] = 1

    return [Q_table_hard, Q_table_soft, Q_table_split]

