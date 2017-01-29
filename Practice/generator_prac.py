CORRECT_COMBO = (3, 6, 1)

def combo_gen():
    for c1 in range(10):
        for c2 in range(10):
            for c3 in range(10):
                yield(c1, c2, c3)

for (c1, c2, c3) in combo_gen():
    if (c1, c2, c3) == CORRECT_COMBO:
        print 'Found the combo: {}'.format((c1, c2, c3))