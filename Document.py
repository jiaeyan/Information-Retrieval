'''
Created on April 20, 2017
@author: jiajieyan
'''
# A class representing the instance characteristics for the NB model.

class Document():
    
    def __init__(self, data, label):
        self.label = label
        self.features = self.process_feature(data)
        
    def process_feature(self, data):
        '''A bag-of-word approach to process raw data, more fancy methods can be applied.
           Default setting is Multinomial, set() the result if a Bernoulli is preferred. 
        '''
        return data.split()  