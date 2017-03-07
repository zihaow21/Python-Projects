fileName = "/home/zwan438/Desktop/eng.list"
new_fileName = "/home/zwan438/Desktop/names.list"
new_new_fileName = "/home/zwan438/Desktop/name.list"
with open(fileName, 'r') as f:
    data = f.readlines()

with open(new_fileName, 'w') as f:
    for item in data:
        item = item.split()
        length = len(item)
        if length > 2:
            for i in range(length-2):
                item[1] += " " + item[i+2]
        print >> f, item[1]

with open(new_fileName, 'r') as f:
    data = f.readlines()
    length = len(data)
    c = 0
    while length > 0:
        item1 = data[c].lower()
        item2 = data[c+1].lower()
        if item1 == item2:
            data.pop(c + 1)
            length -= 1
        else:
            c += 1


with open(new_new_fileName, 'w') as f:
    for i in range(len(data)):
        print >> f, data[i]
