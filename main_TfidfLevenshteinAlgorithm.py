from surprise import AlgoBase
from surprise import PredictionImpossible
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from surprise import Dataset
from surprise.model_selection import train_test_split

import math
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import AlgoBase, PredictionImpossible

class TfidfLevenshteinAlgorithm(AlgoBase):
    def __init__(self, sim_options={}):
        super().__init__(sim_options=sim_options)
        self.vectorizer = TfidfVectorizer()

    def fit(self, trainset):
        super().fit(trainset)
        item_descriptions = [trainset.to_raw_iid(item_id) for item_id in trainset.all_items()]
        item_descriptions = ['d' + str(item_description) for item_description in item_descriptions]
        self.item_tfidf_matrix = self.vectorizer.fit_transform(item_descriptions)
        return self

    def get_best_items(self, user, item):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible('User or item is unknown.')

        item_description = self.trainset.to_raw_iid(item)
        item_tfidf_vector = self.vectorizer.transform([item_description])
        cosine_similarities = self.item_tfidf_matrix.dot(item_tfidf_vector.T).toarray()

        k = self.sim_options['k']
        top_k_items = cosine_similarities.argsort(axis=0)[-k:]
        levenshtein_distances = [Levenshtein.distance(item_description, self.trainset.to_raw_iid(item_id[0]))
                                 for item_id in top_k_items]

        best_items = []
        for i in range(k):
            item_id = top_k_items[i]
            distance = levenshtein_distances[i]
            similarity = math.exp(-distance)

            best_items.append((item_id, similarity))

        best_items.sort(key=lambda x: x[1], reverse=True)

        top_3_items = [item_id for item_id, _ in best_items[:3]]

        return top_3_items