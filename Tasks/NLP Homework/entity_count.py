import re

def entities(graph):
    count = 0
    start_re = r"B-.+"
    inside_re = r"I-.+"
    last_re = r"L-.+"
    start, inside, last = False, False, False
    groups = []
    group = []
    idx = 0
    for i, pos in enumerate(graph):
        if pos == str('O'):
            start, inside, last = False, False, False
            if len(group) != 0:
                group = []
            continue
        else:
            if re.search(start_re, pos):
                start, inside, last = True, False, False
                if len(group) != 0:
                    group = []
                    group.append(pos)
                else:
                    idx = 0
                    idx += i
                    group.append(pos)
                    groups.append((idx, group))
                count += 1
            elif re.search(inside_re, pos):
                if start and last == False:
                    group.append(pos)
                    continue
                else:
                    start, inside, last = False, True, False
                    idx = 0
                    idx += i
                    group.append(pos)
                    groups.append((idx, group))
                    group = []
                    count += 1
            elif re.search(last_re, pos):
                if start == True:
                    group.append(pos)
                    last = True
                    continue
                else:
                    start, inside, last = False, False, True
                    if len(group) != 0:
                        groups.append((idx, group))
                        idx = 0
                        idx += i
                        group = []
                    else:
                        idx = 0
                        idx += i
                        group.append(pos)
                        groups.append((idx, group))
                        group = []
                    count += 1
            else:
                if len(group) != 0:
                    idx = 0
                    idx += i
                    group = []
                    group.append(pos)
                    groups.append((idx, group))
                    group = []
                else:
                    idx = 0
                    idx += i
                    group.append(pos)
                    groups.append((idx, group))
                    group = []
                count += 1
                start, inside, last = False, False, False
    return count, groups

if __name__ == "__main__":
    graph1 = ['U-LOC', 'O', 'B-LOC', 'L-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'L-MISC', 'O', 'U-MISC', 'O', 'O', 'O', 'O', 'O', 'O']
    graph2 = ['U-LOC', 'O', 'B-LOC', 'L-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'L-MISC', 'O', 'B-ORG', 'I-MISC', 'O', 'O', 'O', 'O', 'O']
    print(entities(graph1))
    print(entities(graph2))