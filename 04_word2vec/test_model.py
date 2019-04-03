from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('wiki_zh.model')

testwords = ['苹果', '数学', '学术', '白痴', '篮球']
for i in range(5):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print(testwords[i])
    print(res)
