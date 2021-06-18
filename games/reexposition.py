import numpy as np
import random
import metrics
import copy
import math

class Reexposition_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        self.np_rng = np.random.default_rng()
        pass

    def play(self, recommendations, recommendation_strenghs, items, users, SalesHistory, controlId,
             a, b, c, pi, num_particles, num_generations):
        # print("OLD RECOMMENDATIONS",recommendation_strenghs)
        new_recommendations     = {}
        exposures               = []
        exposure_factors        = pi
        new = {}
        for user in range(len(users.activeUserIndeces)):
            exposure = np.zeros(len(recommendation_strenghs[user]))
            for exposure_factor in range(len(exposure_factors)):
                exposure[exposure_factor] = exposure_factors[exposure_factor]
            exposures.append(exposure)

        optimized_exposure = self.optimize_exposure(items, users, SalesHistory, controlId, exposure_factors,
                                                    recommendations, num_particles, self.number_of_recommendations,
                                                    num_generations, recommendation_strenghs, a, b, c)

        updated_probabilities = self.update_probabilities(users.activeUserIndeces, optimized_exposure, recommendation_strenghs)

        # normalize?

        #print("update_probabilities", updated_probabilities)

        # sort and return best reommendations
        for i in range(len(recommendations)):
            zipped_lists = zip(recommendations[i],updated_probabilities[i])
            sorted_zipped_lists = sorted(zipped_lists,key=lambda x: x[1], reverse=True)
            sorted_user_recommendations = [item[0] for item in sorted_zipped_lists]
            filtered_user_recommendations = sorted_user_recommendations[0:self.number_of_recommendations]
            new_recommendations[i] = filtered_user_recommendations
            print(new_recommendations)


            '''Opitmization:
                PSO/other evolutionary algorithm ?
            '''

            ''' reducing the search space: - consider articles that have not been read much first
                                           - maybe search sequentially, considering that the larger values in the pi vectore should have a larger impact on the final result
                                           - the same goes for articles that are highly relevant to the user (i.e. are recommender system output is very high)
                                           -> search space might be reduced if we search more intensively along some kind of pareto line reconciling these three variables.
            '''
        for i in range(len(recommendations)):
            new[i] = recommendations[i][0:self.number_of_recommendations]

        if new_recommendations==new:
            print("wrong")
        return new_recommendations

    def optimize_exposure(self, items, users, sales_history, controlId, exposure_set, user_recommendations, n_particles,
                          number_of_recommendations, number_of_generations,recommendation_strengths, a, b, c):
        #print("optimize_exposure",sales_history)

        # initialize population
        particles               = []
        best_for_particles      = []
        best_score_per_particle = []
        velocities              = []
        best_neighbour          = None
        best_score              = 0
        a                       = a
        b                       = b
        c                       = c
        a_decay                 = ( a*5/6 )/(number_of_generations)
        #print(exposure_set)

        max_values_per_user     = []
        for user in range(len(user_recommendations)):
            max_values_per_user.append(len(user_recommendations[user]))

        for i in range(n_particles):
            #print(f"particle {i}")
            particle = self.np_rng.integers(min(max_values_per_user), size = len(exposure_set) * len(user_recommendations))
            #print(particle)
            self.legalize_position(particle, len(exposure_set), max_values_per_user)
            best_neighbour = particle
            initial_velocity = self.np_rng.integers(2, size = len(exposure_set) * len(user_recommendations)) - 1
            velocities             .append(initial_velocity)
            particles              .append(particle)
            best_for_particles     .append(particle)
            best_score_per_particle.append(0)


        # iterate for each generation
        for g in range(number_of_generations):
            print(f"Generation {g}/{number_of_generations}")
            for p in range(len(particles)):
                # define movement
                v_inert = a * velocities[p]
                v_previous_best = b * (best_for_particles[p] - particles[p]) * self.np_rng.random()
                v_neighbouring_best = c * (best_neighbour - [particles[p]]) * self.np_rng.random()
                new_velocity = (v_inert + v_previous_best + v_neighbouring_best)
                new_velocity = self.limit_velocity(new_velocity.flatten())
                new_position = particles[p] + new_velocity
                new_position = new_position.flatten()
                velocities[p] = new_velocity

                #new_position = np.ndarray.round(new_position)
                # check for illegal positions
                particles[p] = self.legalize_position(new_position, len(exposure_set), max_values_per_user).flatten()

                # formulate pi from particle position:
                exposure_parameters = []
                for user_id in range(len(user_recommendations)):
                    user_exposure = np.zeros(len(user_recommendations[user_id]))

                    for exposure_index in range(len(exposure_set)):
                        user_exposure[round(particles[p][user_id*len(exposure_set) + exposure_index])] = exposure_set[exposure_index]

                    exposure_parameters.append(user_exposure)

                # update recommendation strengths based on particle position
                updated_probabilities = self.update_probabilities(users.activeUserIndeces, exposure_parameters, recommendation_strengths) #TODO: at this point updated probabilities is not sorted
                new_recommendations = {}
                #print(updated_probabilities)

                #print("OLD:", user_recommendations)

                for i in range(len(user_recommendations)):


                    #new_recommendations[i] = filtered_user_recommendations
                    zipped_lists = zip(user_recommendations[i],updated_probabilities[i])
                    sorted_zipped_lists = sorted(zipped_lists,key=lambda x: x[1], reverse=True)
                    sorted_user_recommendations = [item[0] for item in sorted_zipped_lists]
                    filtered_user_recommendations = sorted_user_recommendations[0:self.number_of_recommendations]
                    new_recommendations[i] = filtered_user_recommendations


                # evaluate position
                value = self.evaluate(users, items, sales_history, new_recommendations, controlId)

                # TODO we also really need to make sure that we refer to the correct items with the indeces we get!

                # after evaluation, update the best positions and the best neighbour value
                if value > best_score_per_particle[p]:
                    best_score_per_particle[p] = value
                    best_for_particles[p] = particles[p]

                    if value > best_score:
                        best_score = value

                        best_neighbour = particles[p] # TODO also make this the best neighbour per round!

            a = a - a_decay

        # formulate pi from particle position:
        exposure_parameters = []
        for user_id in range(len(user_recommendations)):
            user_exposure = np.zeros(len(user_recommendations[user_id]))

            for exposure_index in range(len(exposure_set)):
                user_exposure[round(best_neighbour[user_id*len(exposure_set) + exposure_index])] = exposure_set[exposure_index]

            exposure_parameters.append(user_exposure)
        return exposure_parameters

    def legalize_position(self, particle, parameters_per_user, max_values):
        for i in range(len(particle)):
            max_value = max_values[int(math.floor(i/parameters_per_user))] - 0.5

            left = False
            if self.np_rng.random() > 0.5:
                left = True

            while particle[i] <= -0.5:
                left = False
                particle[i] += 2
            while particle[i] >= max_value:
                left = True
                particle[i] -= 2

            if i%parameters_per_user == 0:
                continue
            else:
                illegal = self.check_illegality(parameters_per_user, particle, i)

                while illegal:
                    if left == True:
                        particle[i] -= 1
                    else:
                        particle[i] += 1
                    while particle[i] <= -0.5:
                        left = False
                        particle[i] += 2
                    while particle[i] >= max_value:
                        left = True
                        particle[i] -= 2

                    illegal = self.check_illegality(parameters_per_user, particle, i)
        return particle

    def check_illegality(self, parameters_per_user, particle, current_index):
        is_illegal = False
        for k in range(current_index%parameters_per_user, 0, -1):
            if round(particle[current_index]) == round(particle[current_index - k]):
                is_illegal = True
                return is_illegal
        return is_illegal

    def limit_velocity(self, velocity):
        for i in range(len(velocity)):
            if velocity[i] > 2:
                velocity[i] = 2
            elif velocity[i] < -2:
                velocity[i] = -2
        return velocity

    def evaluate(self, users, items, sales_history, user_recommendations, controlId):
        ### from the metrics
        sales_history_old = sales_history.copy()
        sales_history_new = sales_history.copy()
        prior_recommendations = np.copy(items.hasBeenRecommended)
        awareness = copy.deepcopy(users.Awareness)
        for user in users.activeUserIndeces:
            Rec=np.array([-1])

            if user not in user_recommendations.keys():
                self.printj(" -- Nothing to recommend -- to user ",user)
                continue
            Rec = user_recommendations[user]
            prior_recommendations[Rec] = 1
            awareness[user, Rec] = 1

            # If recommended but previously purchased, minimize the awareness
            awareness[user, np.where(sales_history_old[user,Rec]>0)[0] ] = 0

        for user in users.activeUserIndeces:
            Rec=np.array([-1])


            if user not in user_recommendations.keys():
                self.printj(" -- Nothing to recommend -- to user ",user)
                continue
            Rec = user_recommendations[user]

            indecesOfChosenItems,indecesOfChosenItemsW =  users.choiceModule(Rec,
                                                                            awareness[user,:],
                                                                            controlId[user,:],
                                                                            users.sessionSize(),)
            sales_history_new[user, indecesOfChosenItems] += 1

        metric = metrics.metrics(sales_history_old, user_recommendations, items.ItemsFeatures, items.ItemsDistances, sales_history_new)

        return metric["EPC"]


    def update_probabilities(self, activeUserIndeces, optimized_exposure, recommendation_strenghs):
        updated_probabilities = {}
        for u in range(len(activeUserIndeces)):
            probability_update = optimized_exposure[u] * np.array(recommendation_strenghs[u])
            updated_probabilities[u] = probability_update
        return updated_probabilities
