import numpy as np
import random

class Reexposition_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, items, users, recommendations, recommendation_strenghs):
        new_recommendations = {}
        exposures = []

        for user in range(len(users.activeUserIndeces)):
            exposure = np.zeros(len(recommendation_strenghs[user]))
            exposure[0] = 1.2
            exposure[1] = 1.1
            exposure[2] = 1.1
            exposure[3] = 1.01
            exposure[4] = 1.01
            exposures.append(exposure)

        optimized_exposure = optimize_exposure(exposure, user_recommendations)

        # apply exposure
        #temp = [list(recommendation_strenghs[key]) for key in recommendation_strenghs]
        #old_probabilities = np.array(temp)

        #updates_probabilities = np.dot(exposure.T, old_probabilities)

        updated_probabilities = {}
        for u in range(len(users.activeUserIndeces)):
            probability_update = optimized_exposure[u] * np.array(recommendation_strenghs[u])
            updated_probabilities[u] = probability_update

        # normalize?


        # sort and return best reommendations
        for i in range(len(recommendations)):
            user_recommendations = np.array([recommendations[i], updated_probabilities[i]])

            sorted_user_recommendations = user_recommendations[user_recommendations[:,1].argsort()]

            filtered_user_recommendations = sorted_user_recommendations[1, 0:self.number_of_recommendations]

            new_recommendations[i] = [int(i) for i in filtered_user_recommendations.tolist()]

            '''Opitmization:
                PSO/other evolutionary algorithm ?
            '''

            ''' reducing the search space: - consider articles that have not been read much first
                                           - maybe search sequentially, considering that the larger values in the pi vectore should have a larger impact on the final result
                                           - the same goes for articles that are highly relevant to the user (i.e. are recommender system output is very high)
                                           -> search space might be reduced if we search more intensively along some kind of pareto line reconciling these three variables.
            '''


            # sorted_user_recommendations = user_recommendations[user_recommendations[:,1].argsort()]

            # filtered_user_recommendations = sorted_user_recommendations[1, 0:self.number_of_recommendations]

            # new_recommendations[i] = [int(i) for i in filtered_user_recommendations.tolist()]

        return new_recommendations

    def optimize_exposure(self, exposure_set, user_recommendations, n_particles, number_of_recommendations, number_of_generations):
        # initialize population
        particles               = []
        best_for_particles      = []
        best_score_per_particle = []
        velocities              = []
        best_neighbour          = None
        best_neighbour_score    = 0
        a_decay                 = 0.00001
        a                       = 2
        b                       = 2
        c                       = 2



        for i in range(len(n_particles)):
            particle = np.random.randint(number_of_recommendations, size = len(exposure_set) * len(user_recommendations))
            self.legalize_position(particle, len(exposure_set), number_of_recommendations)
            best_neighbour = particle
            initial_velocity = np.random.randint(2, size = len(exposure_set) * len(user_recommendations)) - 1
            velocities             .append(initial_velocity)
            particles              .append(particle)
            best_for_particles     .append(particle)
            best_score_per_particle.append(0)

        # iterate for each generation

        for g in range(number_of_generations):
            for p in len(particles):

                # define movement
                v_inert = a * velocities[p]
                v_previous_best = 2 * (best_for_particles[p] - particle[p]) * random.random()
                v_neighbouring_best = 2 * (best_neighbour - [particle[p]]) * random.random()
                new_position = particle[p] + (v_inert + v_previous_best + v_neighbouring_best)

                # check for illegal positions
                particle[p] = self.legalize_position(new_position, len(exposure_set), number_of_recommendations)

                # evaluate position
                #(needs discretization to the items as well)
                # we also really need to make sure that we refer to the correct items with the indeces we get!
                #after evaluation, update the best positions and the best neighbour value

        return #TODO after the last generation, the best neighbouring particle should correspond to the best solution we found

    def legalize_position(self, particle, parameters_per_user, max_value):
        for i in range(len(particle)):
            if i%parameters_per_user == 0:
                continue
            else:
                left = False
                if random.rand() > 0.5:
                    left = True

                illegal = self.check_illegality(parameters_per_user, particle, i)

                while illegal:
                    if left == True:
                        particle[i] -= 1
                    else:
                        particle[i] += 1
                    if particle[i] <= -0.5:
                        left = False
                    if particle[i] >= max_value + 0.5:
                        left = True
                    illegal = self.check_illegality(parameters_per_user, particle, i)

    def check_illegality(self, parameters_per_user, particle, current_index):
        is_illegal = False
        for k in range(parameters_per_user - current_index%parameters_per_user, 0, -1):
            if round(particle[current_index]) == round(particle[current_index - k]):
                is_illegal = True

        return is_illegal
"""
                 |i1|i2|i3|i4|i5|i6|
            0.2u1  x
            0.1u1     x
            0.1u1        x
            0.2u2  x
            0.1u2           x
"""