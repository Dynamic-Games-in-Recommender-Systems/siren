import numpy as np
import random
import metrics

class Reexposition_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, recommendations, recommendation_strenghs, items, users, SalesHistory, controlId):
        new_recommendations     = {}
        exposures               = []
        exposure_factors        = [1.8,
                                   1.2,
                                   1.2,
                                   1.2,
                                   1.2,
                                   1.05,
                                   1.05,
                                   1.05,
                                   1.05,
                                   1.05,
                                   1.05,
                                   1.05,
                                   1.05]

        for user in range(len(users.activeUserIndeces)):
            exposure = np.zeros(len(recommendation_strenghs[user]))
            for exposure_factor in range(len(exposure_factors)):
                exposure[exposure_factor] = exposure_factors[exposure_factor]
            # exposure[0] = 1.05
            # exposure[1] = 1.05
            # exposure[2] = 1.05
            # exposure[3] = 1.05
            # exposure[4] = 1.05
            # exposure[5] = 1.05
            # exposure[6] = 1.05
            # exposure[7] = 1.05
            # exposure[8] = 1.2
            # exposure[9] = 1.2
            # exposure[10] = 1.2
            # exposure[11] = 1.2
            # exposure[12] = 1.8
            exposures.append(exposure)

        optimized_exposure = self.optimize_exposure(items, users, SalesHistory, controlId, exposure_factors, recommendations, 8, self.number_of_recommendations, 20, recommendation_strenghs)

        # apply exposure
        #temp = [list(recommendation_strenghs[key]) for key in recommendation_strenghs]
        #old_probabilities = np.array(temp)


        #updates_probabilities = np.dot(exposure.T, old_probabilities)

        updated_probabilities = self.update_probabilities(users.activeUserIndeces, optimized_exposure, recommendation_strenghs)

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
        print("PLAY RETURN",new_recommendations)
        return new_recommendations

    def optimize_exposure(self, items, users, sales_history, controlId, exposure_set, user_recommendations, n_particles, number_of_recommendations, number_of_generations,recommendation_strengths):
        print("optimize_exposure",sales_history)

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

        for i in range(n_particles):

            particle = np.random.randint(number_of_recommendations, size = len(exposure_set) * len(user_recommendations))
            print(particle)
            self.legalize_position(particle, len(exposure_set), number_of_recommendations)
            best_neighbour = particle
            best_score     = 0
            initial_velocity = np.random.randint(2, size = len(exposure_set) * len(user_recommendations)) - 1
            velocities             .append(initial_velocity)
            particles              .append(particle)
            best_for_particles     .append(particle)
            best_score_per_particle.append(0)


        # iterate for each generation
        for g in range(number_of_generations):
            for p in range(len(particles)):

                # define movement
                v_inert = a * velocities[p]
                v_previous_best = b * (best_for_particles[p] - particles[p]) * random.random()
                v_neighbouring_best = c * (best_neighbour - [particles[p]]) * random.random()
                new_position = particles[p] + (v_inert + v_previous_best + v_neighbouring_best)

                new_position = np.ndarray.round(new_position)
                # check for illegal positions
                particles[p] = self.legalize_position(new_position, len(exposure_set), number_of_recommendations)

                # formulate pi from particle position:
                exposure_parameters = []
                for user_id in range(len(user_recommendations)):
                    user_exposure = np.zeros(len(user_recommendations[user_id]))

                    for exposure_index in range(len(exposure_set)):
                        user_exposure[round(particle[user_id*len(exposure_set) + exposure_index])] = exposure_set[exposure_index]

                    exposure_parameters.append(user_exposure)

                # update recommendation strengths based on particle position
                updated_probabilities = self.update_probabilities(users.activeUserIndeces, exposure_parameters, recommendation_strengths)
                new_recommendations = {}
                for i in range(len(user_recommendations)):

                    recommendations = np.array([user_recommendations[i], updated_probabilities[i]])

                    sorted_user_recommendations = recommendations[recommendations[:,1].argsort()]

                    filtered_user_recommendations = sorted_user_recommendations[1, 0:self.number_of_recommendations]
                    new_recommendations[i] = [int(i) for i in filtered_user_recommendations.tolist()]

                # evaluate position
                value = self.evaluate(users, items, sales_history, new_recommendations, controlId)

                # TODO we also really need to make sure that we refer to the correct items with the indeces we get!

                # after evaluation, update the best positions and the best neighbour value
                if value > best_score_per_particle[p]:
                    best_score_per_particle[p] = value
                    if value > best_score:
                        best_score = value

        # formulate pi from particle position:
        exposure_parameters = []
        for user_id in range(len(user_recommendations)):
            user_exposure = np.zeros(len(user_recommendations[user_id]))

            for exposure_index in range(len(exposure_set)):
                user_exposure[round(particle[user_id*len(exposure_set) + exposure_index])] = exposure_set[exposure_index]

            exposure_parameters.append(user_exposure)

        return exposure_parameters

    def legalize_position(self, particle, parameters_per_user, max_value):
        for i in range(len(particle)):
            if i%parameters_per_user == 0:
                continue
            else:
                left = False
                if random.random() > 0.5:
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
        return particle

    def check_illegality(self, parameters_per_user, particle, current_index):
        is_illegal = False
        for k in range(current_index%parameters_per_user, 0, -1):
            if round(particle[current_index]) == round(particle[current_index - k]):
                is_illegal = True
                return is_illegal
        return is_illegal

    def evaluate(self, users, items, sales_history, user_recommendations, controlId):
        # calculate awareness#
        """ # potentially simplified awareness

        for user in users.activeUserIndeces:
            exposed_items = user_recommendations[user]

            awareness = users.Awareness.copy()
            users.Awareness[user, exposed_items] = 1
            users.Awareness[user, np.where(sales_history[user,exposed_items]>0)[0] ] = 0 # If recommended but previously purchased, minimize the awareness




            indecesOfChosenItems,indecesOfChosenItemsW =  self.U.choiceModule(Rec,
                                                                    self.U.Awareness[user,:], # need to set to one for all selected items
                                                                    self.D[user,:], # the heck is that?
                                                                    self.U.sessionSize(), #this should just be the number of selected items
                                                                    control = self.algorithm=="Control") # we skip this
        """

        ### from the metrics
        sales_history_old = sales_history.copy()
        sales_history_new = sales_history.copy()
        for user in users.activeUserIndeces:
            Rec=np.array([-1])

            if user not in user_recommendations.keys():
                self.printj(" -- Nothing to recommend -- to user ",user)
                continue
            Rec = user_recommendations[user]
            items.hasBeenRecommended[Rec] = 1
            users.Awareness[user, Rec] = 1

                # If recommended but previously purchased, minimize the awareness
            users.Awareness[user, np.where(sales_history_new[user,Rec]>0)[0] ] = 0

        for user in users.activeUserIndeces:
            Rec=np.array([-1])


            if user not in user_recommendations.keys():
                self.printj(" -- Nothing to recommend -- to user ",user)
                continue
            Rec = user_recommendations[user]

            indecesOfChosenItems,indecesOfChosenItemsW =  users.choiceModule(Rec,
                                                                            users.Awareness[user,:],
                                                                            controlId[user,:],
                                                                            users.sessionSize(),)
            sales_history_new[user, indecesOfChosenItems] += 1

        metric = metrics.metrics(sales_history_old, user_recommendations, items.ItemsFeatures, items.ItemsDistances, sales_history_new)
        print("evaluate-old",sales_history)
        print("evaluate-new",sales_history_new)
        print("evaluate-diff",sales_history_new-sales_history_old)

        print("EPC", metric)
        return metric


    def update_probabilities(self, activeUserIndeces, optimized_exposure, recommendation_strenghs):
        updated_probabilities = {}
        for u in range(len(activeUserIndeces)):
            probability_update = optimized_exposure[u] * np.array(recommendation_strenghs[u])
            updated_probabilities[u] = probability_update
        return updated_probabilities
