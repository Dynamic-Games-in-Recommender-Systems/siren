import numpy as np
import random

class Reexposition_game:
    def __init__(self, number_of_recommendations):
        self.number_of_recommendations = number_of_recommendations
        pass

    def play(self, recommendations, recommendation_strenghs, items, users, SalesHistory, controlId):
        new_recommendations = {}
        exposures = []
        SalesHistory1 = SalesHistory.copy()

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

            return new_recommendations

    def optimize_exposure(self, users, sales_history, exposure_set, user_recommendations, n_particles, number_of_recommendations, number_of_generations):
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
                v_previous_best = b * (best_for_particles[p] - particle[p]) * random.random()
                v_neighbouring_best = c * (best_neighbour - [particle[p]]) * random.random()
                new_position = particle[p] + (v_inert + v_previous_best + v_neighbouring_best)

                # check for illegal positions
                particle[p] = self.legalize_position(new_position, len(exposure_set), number_of_recommendations)

                # formulate pi from particle position:
                exposure_parameters = []
                for user_id in range(len(user_recommendations)):
                    user_exposure = np.zeros(len(user_recommendations[user_id]))

                    for exposure_index in range(len(exposure_set)):
                        user_exposure[round(particle[user_id*len(exposure_set) + exposure_index])] = exposure_set[exposure_index]

                    exposure_parameters.append(user_exposure)

                # update recommendation strengths based on particle position
                updated_probabilities = self.update_probabilities(users.activeUserIndeces, exposure_parameters, user_recommendations)

                # evaluate position
                value = self.evaluate(users, sales_history, exposure_parameters, updated_probabilities, n_particles, number_of_recommendations)

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

    def evaluate(self, users, items, sales_history, user_recommendations, metrics, controlId):
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
        for user in users.activeUserIndeces:
            Rec=np.array([-1])

            if user not in user_recommendations.keys():
                self.printj(" -- Nothing to recommend -- to user ",user)
                continue
            Rec = user_recommendations[user]
            items.hasBeenRecommended[Rec] = 1
            users.Awareness[user, Rec] = 1

                # If recommended but previously purchased, minimize the awareness
            users.Awareness[user, np.where(sales_history[user,Rec]>0)[0] ] = 0

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
            sales_history[user, indecesOfChosenItems] += 1

        metric = metrics.metrics(sales_history_old, Rec, items.ItemsFeatures, items.ItemsDistances, sales_history)
        return metric


    def update_probabilities(self, activeUserIndeces, optimized_exposure, recommendation_strenghs):
        updated_probabilities = {}
        for u in range(len(activeUserIndeces)):
            probability_update = optimized_exposure[u] * np.array(recommendation_strenghs[u])
            updated_probabilities[u] = probability_update
        return updated_probabilities

        """
        Similarity = -self.k*np.log(distanceToItems)
        V = Similarity.copy()

        if not control:
            # exponential ranking discount, from Vargas
            for k, r in enumerate(Rec):
                V[r] = Similarity[r] + self.delta*np.power(self.beta,k)

        # Introduce the stochastic component
        E = -np.log(-np.log([random.random() for v in range(len(V))]))
        U = V + E
        sel = np.where(w==1)[0]

        # with stochastic
        selected = np.argsort(U[sel])[::-1]

        # without stochastic
        selectedW = np.argsort(V[sel])[::-1]
        return sel[selected[:sessionSize]],sel[selectedW[:sessionSize]]
        """


        """for user in self.U.activeUserIndeces:
                        Rec = recommendations[user]

                    indecesOfChosenItems,indecesOfChosenItemsW =  self.U.choiceModule(Rec,
                                                                                      self.U.Awareness[user,:],
                                                                                      self.D[user,:],
                                                                                      self.U.sessionSize(),
                                                                                      control = self.algorithm=="Control")

                    # Add item purchase to histories
                    self.SalesHistory[user, indecesOfChosenItems] += 1

                    # Compute new user position
                    if self.algorithm is not "Control" and len(indecesOfChosenItems)>0:
                        self.U.computeNewPositionOfUser(user, self.I.Items[indecesOfChosenItems])"""
