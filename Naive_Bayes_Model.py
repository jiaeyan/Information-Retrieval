'''
Created on April 20, 2017

@author: jiajieyan
'''
import numpy as np

# A numpy-matrix based Multinomial/Bernoulli Naive Bayes Model, with add-0.01 smoothing.

class NaiveBayes():
    
    def train(self, data):  # data format: [d1, d2, d3, ...]
        '''Get the feature type and class type to decide the shapes of matrices.'''
        F, C = set(), set() # F: seen feature set; C: seen class set
        for d in data:
            C.add(d.label)
            F = F.union(d.features)
        self.F = {f:i for i, f in enumerate(F)} # a dict to record feature and its ID in matrix
        self.C = {}                             # a two-way dict to record label and its ID in matrix
        for j, c in enumerate(C):
            self.C[c] = j 
            self.C[j] = c 
        self.P, self.MLE = self.count(data, np.zeros(len(C)), np.zeros((len(F), len(C))) + 0.01)
    
    def count(self, data, P, MLE):
        '''Fill in the matrices and normalize.
           P: prior dict of class
           MLE: maximum likelihood estimation of feature
        '''
        for d in data:
            c = d.label
            P[self.C[c]] += 1
            for f in d.features:
                MLE[self.F[f]][self.C[c]] += 1
        MLE = np.log(MLE / (P + 0.01 * len(MLE)))
        P = np.log(P / len(data))
        return P, MLE
    
    def get_vec(self, d):
        '''Make a feature vector of the given doc.'''
        d_vec = np.zeros(len(self.F))
        for f in d.features:
            if f in self.F: d_vec[self.F[f]] += 1 # ignore OOV/unkown words
        return d_vec
        
    def classify(self, d):
        '''Inference the best class for the given doc.'''
        d_vec = self.get_vec(d)
        return max([(np.dot(d_vec, self.MLE[:, c]) + self.P[c], self.C[c]) for c in range(self.MLE.shape[1])])[1]