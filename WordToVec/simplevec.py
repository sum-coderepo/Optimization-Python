from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'D:\\GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
result = model.most_similar(positive=['student', 'college'], topn=1)
print(result)
result = model.most_similar(positive=['run','fast','animal'], topn=1)
print(result)
result = model.most_similar(positive=['speed','wheel'],negative=['slow'], topn=1)
print(result)