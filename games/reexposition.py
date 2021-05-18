class Reexposition_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, items, users, recommendations, recommendation_strenghs):
        new_recommendations = {}
        for i in range(len(recommendations)):
            user_recommendations = np.array([recommendations[i], recommendation_strenghs[i]])

        return new_recommendations