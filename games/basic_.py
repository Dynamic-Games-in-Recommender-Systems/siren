import numpy as np

class Basic_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, items, users, recommendations, recommendation_strenghs):
        new_recommendations = {}
        for i in range(len(recommendations)):
            user_recommendations = np.array([recommendations[i], recommendation_strenghs[i]])

            sorted_user_recommendations = user_recommendations[user_recommendations[:,1].argsort()]

            filtered_user_recommendations = sorted_user_recommendations[1, 0:self.number_of_recommendations]

            new_recommendations[i] = [int(i) for i in filtered_user_recommendations.tolist()]
        return new_recommendations