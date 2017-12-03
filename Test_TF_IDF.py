from nltk.corpus import brown  

def read():
    data = []
    for fid in brown.fileids():
        doc = Document(fid, brown.words(fid), brown.sents(fid))
        data.append(doc)
    return data

data = read()[:100]
ti = TFIDF(data)
name = brown.fileids()[15]
kws = ti.getKeyWords(5, name)
print(kws)
print()

docs = ti.getDocs(['document', 'issue'], 5)
for doc in docs:
    print(brown.words(doc))
print()

simdocs = ti.simDocs(5, 5, name)
for doc in simdocs:
    print(brown.words(doc))
print()

doc = Document(name, brown.words(name), brown.sents(name))
summary = ti.summarize(5, doc=doc)
for sent in summary:
    print(sent)
