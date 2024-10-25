import math as math

# c*functions(x-a) + b
weights = [[
    1,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4
], [
    0.3,
    0.2,
    0.1,
    0.05,
]]
directions = [
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
]

def evaluate(scores):
    pos = []
    neg = []
    for i in range(len(scores)):
        if directions[i] >= 0:
            pos.append(scores[i])
        else:
            neg.append(scores[i])
    pos = sorted(pos, reverse=True)
    neg = sorted(neg, reverse=True)
    res = 0
    for i in range(len(pos)):
        res += weights[0][i] * pos[i]
    for i in range(len(neg)):
        res += weights[1][i] * neg[i]
    return res