class Padding(object):
    def __init__(self):
        pass

    def padding(self, data, data_length):
        max_len = max(data_length)

        for i in range(len(data)):
            data[i] += [0] * (max_len - len(data[i]))

        return data
