import fasttext as ftt


class FastText(object):
    def __init__(self, model_dir):
        self.model = ftt.load_model(model_dir)

    # def modelCheck(self, word):
    #     vector = self.model[word]
    #
    #     return vector