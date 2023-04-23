from surprise import AlgoBase
from surprise import PredictionImpossible
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from surprise import Dataset
from surprise.model_selection import train_test_split


class TfidfLevenshteinAlgorithm(AlgoBase):
    def init(self, sim_options={}):
        AlgoBase.init(self, sim_options=sim_options)
        self.vectorizer = TfidfVectorizer()

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        item_descriptions = [trainset.to_raw_iid(item_id) for item_id in
                             trainset.all_items()]
        item_descriptions = ['d' + str(item_description) for item_description
                             in item_descriptions]
        self.item_tfidf_matrix = self.vectorizer.fit_transform(
            item_descriptions)
        return self

    def estimate(self, user, item):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(
                item)):
            raise PredictionImpossible('User or item is unknown.')
        item_description = self.trainset.to_raw_iid(item)
        item_tfidf_vector = self.vectorizer.transform([item_description])
        cosine_similarities = self.item_tfidf_matrix.dot(
            item_tfidf_vector.T).toarray()
        k = self.sim_options['k']
        top_k_items = cosine_similarities.argsort(axis=0)[-k:]
        user_input = self.trainset.to_raw_uid(user)
        item_names = [self.trainset.to_raw_iid(item_id[0]) for item_id in
                      top_k_items]
        levenshtein_distances = [Levenshtein.distance(user_input, item_name)
                                 for item_name in item_names]
        rating_sum = 0
        weight_sum = 0
        for i in range(k):
            item_id = top_k_items[i]
            rating = self.trainset.ur[user][item_id].r_ui
            distance = levenshtein_distances[i]
            weight = math.exp(-distance)
            rating_sum += weight * rating
            weight_sum += weight
        if weight_sum == 0:
            raise PredictionImpossible('No similar items found.')
        est_rating = rating_sum / weight_sum
        return est_rating


