class Reexposition_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, items, users, recommendation_strengh):
        new_recommendations = recommendation_strengh
        return new_recommendations