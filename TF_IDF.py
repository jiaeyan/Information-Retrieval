
'''
This is a class to perform various information retrieval techniques.
TF-IDF algorithm that does keywords extraction, documents searching by keyword list, 
similar documents searching and summarization are all included.

tf = term frequency: the frequency of word i in document j
idf = inverse document frequency: N/df, where df is the frequency of document that has word i, N the total number of documents
tf-idf weighting of the value for word i in document j:
wij =tfij * idfi
Tf-idf thus prefers words that are frequent in the current document j but rare overall in the collection.
we can use term-document matrix to record all info, where row represents the word frequency, column represents the document
'''
from numpy import zeros, sum, log, count_nonzero, argsort
from collections import Counter
from scipy.spatial.distance import cosine
from nltk.corpus import reuters

class TFIDF():
    '''
    D: two way document-id dict
    W: two way word-id dict
    num_D: number of document in training data
    num_W: number of word types in training data
    M: term-document frequency matrix
    #D_KW: a dict mapping dac names in training data to its keywords
    '''
    def __init__(self, data):
        self.D, self.W, self.num_D, self.num_W, M = self.formuate(data)
        self.M = self.train(data, M)
#         self.D_KW = {doc.name:self.getKeyWords(n, doc.name) for doc in data}
        
    def formuate(self, data):
        d, w = set(), set()
        for doc in data:
            d.add(doc.name)
            for f in doc.words:
                w.add(f)
        D = self.makeDict(d)
        W = self.makeDict(w)
        return D, W, len(d), len(w), zeros((len(w), len(d)))
    
    def makeDict(self, d):
        res = {}
        for i, elem in enumerate(d):
            res[elem] = i
            res[i] = elem
        return res
    
    def train(self, data, M):
        for doc in data:
            col = self.D[doc.name]
            for f in doc.words:
                M[self.W[f], col] += 1
        return M
    
    def getKeyWords(self, n, name = None, doc = None):
        '''
        Return the keywords of given doc. Either name or doc must be provided, but not both.
        
        @parameter:
        n: the number of keywords returned
        name: the document's name from training data
        doc: the new document that needs keywords
        
        @variable:
        wf: word frequency dict for this doc
        wc: total number of word tokens in this doc
        wwlist: word weight list
        
        @return:
        A list of n keywords that have the biggest tf-idf values.
        '''
        wf = Counter(doc.words) if doc != None else {self.W[i]:self.M[i, self.D[name]] for i in range(self.num_W)}
        wc = len(doc.words) if doc != None else sum(self.M[:, self.D[name]])      
        wwlist = [(self.wordWeight(tf, wc, w), w) for w, tf in wf.items()]
        wwlist.sort(reverse = True)
        return [pair[1] for pair in wwlist[:n]]
    
    def wordWeight(self, tf, wc, word):
        df = 1 if word not in self.W else count_nonzero(self.M[self.W[word]]) #handle oov in test doc
        return (tf / wc) * log(self.num_D / df)
    
    def getDocs(self, keyWords, n):
        '''
        Return the documents' names that match the given keywords in the training data.
        For each document, sum over all tf-idf values of keywords, get the max ones. 
        
        @parameter:
        keyWords: a list of search words
        n: the number of relative docs returned
        
        @variable:
        kwlist: only consider the keyword in the data
        wc: a vector of sum of tokens in each document in data, len = num_d
        tf: a vector of number of every kw in each document in data, len = num_d
        idf: a vector of idf value for each keyword, len = len(kwlist)
        doc_index: a sorted vector of indices of doc with row: sum of kw values, col: D, from a matrix with row: D, col: kw values        
        
        @return: 
        A list of n most possible documents' names that match the keyword list.
        '''
        kwlist = [word for word in keyWords if word in self.W]
        wc = sum(self.M, axis = 0)
        tf = [self.M[self.W[kw]] for kw in kwlist]
        idf = [log(self.num_D / count_nonzero(self.M[self.W[kw]])) for kw in kwlist]
        doc_index = argsort(sum((tf / wc).transpose() * idf, axis = 1))[::-1][:n]
        return [self.D[index] for index in doc_index]
    
    def simDocs(self, n, m, name = None, doc = None):
        '''
        Return the documents' names in training data that are similar with given document.
        Either document's name or new doc should be provided, but not both.
        Extract same number of keywords from each document, form 2 vectors, compute their cosim.
        
        @parameter:
        name: the document's name from training data
        doc: the new document to be compared
        n: the number of doc names returned
        m: the number of keywords needed for each document
        
        @return: 
        A list of documents' names that are most similar with given doc.
        '''
        res = zeros(self.num_D)
        kws_input = self.getKeyWords(m, name) if name != None else self.getKeyWords(m, doc)
        wf_input = Counter(doc.words) if doc != None else {self.W[i]:self.M[i, self.D[name]] for i in range(self.num_W)}
        wc_input = len(doc.words) if doc != None else sum(self.M[:, self.D[name]]) 
        for doc in range(self.num_D):
            kws_doc = self.getKeyWords(m, self.D[doc])
            wc_doc = sum(self.M[:, doc])
            kws = [kw for kw in set(kws_input + kws_doc) if kw in self.W] #ignore oov
            vec_input = [wf_input[kw]/wc_input for kw in kws]
            vec_doc = [self.M[self.W[kw], doc]/wc_doc for kw in kws]
            cosim = 1 - cosine(vec_input, vec_doc)
            res[doc] = cosim
        doc_index = argsort(res)[::-1][:n]
        return [self.D[index] for index in doc_index]
    
    def summarize(self, n, name = None, doc = None):
        '''
        Return the summarization of given document.
        Compute its keywords, get the first sentences that contain the keywords.
        
        @parameter:
        name: the document's name from training data
        doc: the new document to be compared
        n: the number of keywords returned, also the number of sentences.
        
        @return: 
        A list of sentences that contain the keywords.
        '''
        res = []
        kws = self.getKeyWords(n, name) if name != None else self.getKeyWords(n, doc)
        for kw in kws:
            for sentence in doc.sentences:
                if kw in sentence and sentence not in res: 
                    res.add(sentence)
                    break
        return [' '.join(sentence) for sentence in res]
