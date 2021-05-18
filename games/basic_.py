class Basic_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, items, users, recommendation_strenghs):
        new_recommendations = recommendation_strenghs
        for i in range(1,len(recommendation_strenghs)):
            new_recommendations[i]
        return new_recommendations