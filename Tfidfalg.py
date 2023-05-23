from surprise import AlgoBase, PredictionImpossible
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfAlgorithm(AlgoBase):
    def __init__(self, sim_options={}):
        super().__init__(sim_options=sim_options)
        self.vectorizer = TfidfVectorizer()

    def fit(self, trainset):
        super().fit(trainset)
        item_descriptions = [trainset.to_raw_iid(item_id) for item_id in trainset.all_items()]
        item_descriptions = ['d' + str(item_description) for item_description in item_descriptions]
        self.item_tfidf_matrix = self.vectorizer.fit_transform(item_descriptions)
        return self
    
    def get_best_item(self, user, item):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible('User or item is unknown.')

        item_description = self.trainset.to_raw_iid(item)
        item_tfidf_vector = self.vectorizer.transform([item_description])
        cosine_similarities = self.item_tfidf_matrix.dot(item_tfidf_vector.T).toarray()

        k = self.sim_options['k']
        top_k_items = cosine_similarities.argsort(axis=0)[-k:]

        best_similarity = 0
        best_item = None

        for item_id in top_k_items:
            item_similarity = cosine_similarities[item_id]

            if item_similarity > best_similarity:
                best_similarity = item_similarity
                best_item = item_id

        return best_item
