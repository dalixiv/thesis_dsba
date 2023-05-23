from surprise import AlgoBase, PredictionImpossible
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

class Word2VecAlgorithm(AlgoBase):
    def __init__(self, sim_options={}):
        super().__init__(sim_options=sim_options)
        self.word2vec_model = Word2Vec()
        self.item_embeddings = None

    def fit(self, trainset):
        super().fit(trainset)
        item_descriptions = [trainset.to_raw_iid(item_id) for item_id in trainset.all_items()]
        item_descriptions = [item_description.split() for item_description in item_descriptions]
        self.word2vec_model.build_vocab(item_descriptions)
        self.word2vec_model.train(item_descriptions, total_examples=len(item_descriptions), epochs=10)
        item_embedding = normalize(self.word2vec_model.wv[item_description].reshape(1, -1))
        return self
    
    def get_best_item(self, user, item):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible('User or item is unknown.')

        item_description = self.trainset.to_raw_iid(item)
        item_description = item_description.split()
        item_embedding = self.word2vec_model.wv[item_description]
        item_embedding = item_embedding / np.linalg.norm(item_embedding)

        cosine_similarities = self.item_embeddings.dot(item_embedding.T)

        k = 1
        best_item = cosine_similarities.argsort(axis=0)[-k:]

        return best_item



