
# A class representing the instance characteristics for tf-idf algorithm.

class Document():
    
    def __init__(self, name, words, sentences, label):
        self.name = name
        self.words = self.features(words)
        self.sentences = sentences
        self.label = label
    
    '''Implement self defined feature engineering here.'''
    def features(self, words):
        return words
