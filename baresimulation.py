""" Simulation of online news consumption including recommendations.

A simulation framework  for the visualization and analysis of the effects of different recommenders systems.
This simulation draws mainly on the work of Fleder and Hosanagar (2017). To account for the specificities
of news consumption, it includes both users preferences and editorial priming as they interact in a
news-webpage context. The simulation models two main components: users (preferences and behavior)
and items (article content, publishing habits). Users interact with items and are recommended items
based on their interaction.

Example:

	$ python3 simulation.py

"""

from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2
import random
import pandas as pd
import pickle
from sklearn.mixture import GaussianMixture
import os
import sys, getopt
import copy
import json
import metrics
import matplotlib
import bisect
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
import traceback, sys
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)
from PyQt5.QtWidgets import *
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


totalNumberOfIterationsSimulation = 1

def cdf(weights):

    """ Cummulative density function.

    Used to convert topic weights into probabilities.

    Args:
        weights (list): An array of floats corresponding to weights

    """

    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def selectClassFromDistribution(population, weights):
    """ Given a list of classes and corresponding weights randomly select a class.

    Args:
        population (list): A list of class names e.g. business, politics etc
        weights (list): Corresponding float weights for each class.

    """

    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

def standardize(num, precision = 2):
    """ Convert number to certain precision.

    Args:
        num (float): Number to be converted
        precision (int): Precision either 2 or 4 for now

    """

    if precision == 2:
        return float("%.2f"%(num))
    if precision == 4:
        return float("%.4f"%(num))

def euclideanDistance(A,B):
    """ Compute the pairwise distance between arrays of (x,y) points.

    We use a numpy version which is C++ based for the sake of efficiency.

    """

    #spatial.distance.cdist(A, B, metric = 'euclidean')
    return np.sqrt(np.sum((np.array(A)[None, :] - np.array(B)[:, None])**2, -1)).T


#%% USER CLASS
class Users(object):
    """ The class for modeling the user preferences (users) and user behavior.

    The users object can be passed from simulation to simulation, allowing for
    different recommendation algorithms to be applied on. The default attributes
    correspond to findings reports on online news behavior (mostly Mitchell et
    al 2017,'How Americans encounter, recall and act upon digital news').

    """

    def __init__(self):
        """ The initialization simply sets the default attributes.

        """

        self.seed = 1

        self.totalNumberOfUsers = 200  # Total number of users
        self.percentageOfActiveUsersPI = 1.0 # Percentage of active users per iterations

        self.m = 0.05  # Percentage of the distance_ij covered when a user_i drifts towards an item_j

        # Choice attributes
        self.k = 20
        self.delta = 5
        self.beta = 0.9
        self.meanSessionSize = 6

        # Awareness attributes
        self.theta = 0.07  # Proximity decay
        self.thetaDot = 0.5  # Prominence decay
        self.Lambda = 0.6  # Awareness balance between items in proximity and prominent items
        self.w = 40  # Maximum awareness pool size
        self.Awareness = [] # User-item awareness matrix

        self.Users = []  # User preferences, (x,y) position of users on the attribute space
        self.UsersClass = []  # Users be assigned a class (center at attribute space)
        self.userVarietySeeking = []  # Users' willingness to drift
        self.X = False  # Tracking the X,Y position of users throught the simulation
        self.Y = False

    def generatePopulation(self):
        """ Genererating a population of users (user preferences and variety seeking).

        """

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Position on the attribute space. Uniform, bounded by 1-radius circle
        self.Users = np.random.uniform(-1,1,(self.totalNumberOfUsers,2))
        for i, user in enumerate(self.Users):
            while euclideanDistance([user], [[0,0]])[0][0]>1.1:
                user = np.random.uniform(-1,1,(1,2))[0]
            self.Users[i] = user

        # Variety seeking, willingness to drift. Arbitrary defined
        lower, upper = 0, 1
        mu, sigma = 0.1, 0.03
        X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self.userVarietySeeking = X.rvs(self.totalNumberOfUsers, random_state = self.seed)

        # Users can be assigned a class (most proxiamte attribute center), not currently used.
        #self.UsersClass = [gmm.predict([self.Users[i]*55])[0] for i in range(self.totalNumberOfUsers)]

        self.X = {i:[self.Users[i,0]] for i in range(self.totalNumberOfUsers)}
        self.Y = {i:[self.Users[i,1]] for i in range(self.totalNumberOfUsers)}

    def sessionSize(self):
        """ Draw the session size (amount of items to purchase) of each user at each iteration from a normal distribution.

        Returns:
            int: the session size

        """

        return int(np.random.normal(self.meanSessionSize, 2))

    def subsetOfAvailableUsers(self):
        """ Randomly select a subset of the users.

        """

        self.activeUserIndeces = np.arange(self.totalNumberOfUsers).tolist()
        random.shuffle(self.activeUserIndeces)
        self.activeUserIndeces = self.activeUserIndeces[:int(len(self.activeUserIndeces)*self.percentageOfActiveUsersPI)]
        self.nonActiveUserIndeces = [ i  for i in np.arange(self.totalNumberOfUsers) if i not in self.activeUserIndeces]

    def computeAwarenessMatrix(self, Dij, ItemProminence, activeItemIndeces):
        """ Compute awareness from proximity and prominence (not considering availability, recommendations, history).

        Args:
            Dij (nparray): |Users| x |Items| distance matrix
            ItemProminence (nparray): |Items|-sized prominence vector

        """

        totalNumberOfItems = ItemProminence.shape[0]

        W = np.zeros([self.totalNumberOfUsers,totalNumberOfItems])
        W2 = W.copy() # for analysis purposes
        W3 = W.copy() # for analysis purposes
        for a in self.activeUserIndeces:
            W[a,activeItemIndeces] = self.Lambda*(-self.thetaDot*np.log(1-ItemProminence[activeItemIndeces])) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,activeItemIndeces],2))/self.theta)
            W2[a,activeItemIndeces] = self.Lambda*(-self.thetaDot*np.log(1-ItemProminence[activeItemIndeces]))
            W3[a,activeItemIndeces] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,activeItemIndeces],2))/self.theta)
        R = np.random.rand(W.shape[0],W.shape[1])
        W = R<W
        self.Awareness, self.AwarenessOnlyPopular, self.AwarenessProximity =  W, W2, W3

    def choiceModule(self, Rec, w, distanceToItems, sessionSize, control = False):
        """ Selecting items to purchase for a single user.

        Args:
            Rec (list): List of items recommended to the user
            w (nparray): 1 x |Items| awareness of the user
            distanceToItems (nparray): 1 x |Items| distance of the user to the items
            sessionSize (int): number of items that the user will purchase

        Returns:
             param1 (list): List of items that were selected including the stochastic component
             param2 (list): List of items that were selected not including the stochastic component

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

    def computeNewPositionOfUser(self, user, ChosenItems):
        """ Compute new position of a user given their purchased item(s).

        Args:
            user (int): Index of specific user.
            ChosenItems (list): (x,y) position array of items selected by the user.

        """

        for itemPosition in ChosenItems:
            dist =  euclideanDistance([self.Users[user]], [itemPosition])[0]
            p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
            B = np.array(self.Users[user])
            P = np.array(itemPosition)
            BP = P - B
            x,y = B + self.m*(random.random()<p)*BP
            self.Users[user] = [x,y]
        self.X[user].append(x)
        self.Y[user].append(y)

class Items(object):
    """ The class for modeling the items' content (items) and prominence.

    The items object can be passed from simulation to simulation, allowing for
    different recommendation algorithms to be applied on. The default attributes
    correspond to findings reports on online news behavior (mostly Mitchell et
    al 2017,'How Americans encounter, recall and act upon digital news').

    """
    def __init__(self):
        """ The initialization simply sets the default attributes.

        """
        self.seed = 1
        self.numberOfNewItemsPI = 100  # The number of new items added per iteration
        self.totalNumberOfItems = False  # The total number of items (relates to the number of iterations)
        self.percentageOfActiveItems = False

        # Topics, frequency weights and prominence weights. We use topics instead of "classes" here.
        self.topics = ["entertainment","business","politics","sport","tech"]
        self.topicsProminence = [] #[0.05,0.07,0.03,0.85,0.01]
        self.topicsFrequency = [] #[0.2, 0.2, 0.2, 0.2, 0.2]

        self.p = 0.1  # Slope of salience decrease function

        self.Items = []  # The items' content (x,y) position on the attribute space
        self.ItemsClass = []  # The items' class corresponds to the most prominent topic
        self.ItemsFeatures = False  # The items' feature vector
        self.ItemsDistances = False  # |Items|x|Items| distance matrix
        self.ItemsOrderOfAppearance = False  # Random order of appearance at each iteration
        self.ItemProminence = False  #  Item's prominence
        self.ItemLifespan = False  # Items' age (in iterations)
        self.hasBeenRecommended = False  # Binary matrix holding whether each items has been recommended

    def generatePopulation(self, totalNumberOfIterations):
        """ Genererating a population of items (items' content and initial prominence).

        """

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Compute number of total items in the simulation
        self.totalNumberOfItems = totalNumberOfIterations*self.numberOfNewItemsPI
        self.percentageOfActiveItems = self.numberOfNewItemsPI/self.totalNumberOfItems

        # Apply GMM on items/articles from the BBC data
        R, S = [5,1,6,7], [5,2,28,28]
        r = int(random.random()*4)
        (X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection'+str(R[r])+'.pkl','rb'))
        gmm = GaussianMixture(n_components=5, random_state=S[r]).fit(X)

        # Normalize topic weights to sum into 1 (CBF)
        self.topicsFrequency = [np.round(i,decimals=1) for i in self.topicsFrequency/np.sum(self.topicsFrequency)]

        # Generate items/articles from the BBC data projection
        samples_, classes_ = gmm.sample(self.totalNumberOfItems*10)
        for c, category in enumerate(self.topics):
            selection = samples_[np.where(classes_ == c)][:int(self.topicsFrequency[c]*self.totalNumberOfItems)]
            if len(self.Items) == 0:
                self.Items = np.array(selection)
            else:
                self.Items = np.append(self.Items, selection, axis=0)
            self.ItemsClass+=[c for i in range(len(selection))]
        self.ItemsClass = np.array(self.ItemsClass)
        self.ItemsFeatures = gmm.predict_proba(self.Items)
        self.Items = self.Items/55  # Scale down to -1, 1 range

        # Cosine distance between item features
        self.ItemsDistances = spatial.distance.cdist(self.ItemsFeatures, self.ItemsFeatures, metric='cosine')

        # Generate a random order of item availability
        self.ItemsOrderOfAppearance = np.arange(self.totalNumberOfItems).tolist()
        random.shuffle(self.ItemsOrderOfAppearance)

        # Initial prominence
        self.initialProminceZ0()
        self.ItemProminence = self.ItemsInitialProminence.copy()

        # Lifespan, item age
        self.ItemLifespan = np.ones(self.totalNumberOfItems)

        # Has been recommended before
        self.hasBeenRecommended = np.zeros(self.totalNumberOfItems)

    def prominenceFunction(self, initialProminence, life):
        """ Decrease of item prominence, linear function.

        Args:
            initialProminence (float): The initial prominence of the item
            life (int): The item's age (in iterations)

        Returns:
            param1 (float): New prominence value

        """

        x = life
        y = (-self.p*(x-1)+1)*initialProminence
        return max([y, 0])

    def subsetOfAvailableItems(self,iteration):
        """ Randomly select a subset of the items.

        The random order of appearance has already been defined in ItemsOrderOfAppearance. The function simply
        extends the size of the activeItemIndeces array.

        Args:
            iteration (int): the current simulation iteration

        """

        self.activeItemIndeces =[j for j in self.ItemsOrderOfAppearance[:(iteration+1)*int(self.totalNumberOfItems*self.percentageOfActiveItems)] if self.ItemProminence[j]>0]
        self.nonActiveItemIndeces = [ i  for i in np.arange(self.totalNumberOfItems) if i not in self.activeItemIndeces]

    def updateLifespanAndProminence(self):
        """ Update the lifespan and promince of the items.

        """

        self.ItemLifespan[self.activeItemIndeces] = self.ItemLifespan[self.activeItemIndeces]+1

        for a in self.activeItemIndeces:
            self.ItemProminence[a] = self.prominenceFunction(self.ItemsInitialProminence[a],self.ItemLifespan[a])

    def initialProminceZ0(self):
        """ Generate initial item prominence based on the topic weights and topic prominence.

        """

        self.topicsProminence = [np.round(i,decimals=2) for i in self.topicsProminence/np.sum(self.topicsProminence)]
        counts = dict(zip(self.topics, [len(np.where(self.ItemsClass==i)[0]) for i,c in enumerate(self.topics) ]))
        items = len(self.ItemsClass)
        population = self.topics

        # Chi square distribution with two degrees of freedom. Other power-law distributions can be used.
        df = 2
        mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
        x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), items)
        rv = chi2(df)

        Z = {}
        for c in self.topics: Z.update({c:[]})

        # Assign topic to z prominence without replacement
        for i in rv.pdf(x):
            c = selectClassFromDistribution(population, self.topicsProminence)
            while counts[c]<=0:
                c = selectClassFromDistribution(population, self.topicsProminence)
            counts[c]-=1
            Z[c].append(i/0.5)

        self.ItemsInitialProminence = np.zeros(self.totalNumberOfItems)
        for c, category in enumerate(self.topics):
            indeces = np.where(self.ItemsClass==c)[0]
            self.ItemsInitialProminence[indeces] = Z[category]

class Recommendations(object):
    def __init__(self):
        self.outfolder = ""
        self.SalesHistory = []
        self.U = []
        self.I = []
        self.algorithm = False
        self.n = 5

    def setData(self, U, I, algorithm, SalesHistory):
        self.U, self.I, self.algorithm, self.SalesHistory = U, I, algorithm, SalesHistory

    def exportToMMLdocuments(self):
        """ Export users' features, items' content and user-item purchase history for MyMediaLite.

        MyMediaLite has a specific binary input format for user-, item-attributes: the attribute
        either belongs or does not belong to an item or user. To accommodate for that we had to
        take some liberties and convert the user's feature vector and item's feature vector into
        a binary format.


        """

        np.savetxt(self.outfolder + "/users.csv", np.array([i for i in range(self.U.totalNumberOfUsers)]), delimiter=",", fmt='%d')

        F = []
        for user in range(self.SalesHistory.shape[0]):
            purchases = self.SalesHistory[user,:]
            items = np.where(purchases==1)[0]
            userf = self.I.ItemsFeatures[items]
            userfm = np.mean(userf,axis=0)
            userfm = userfm/np.max(userfm)
            feat = np.where(userfm>0.33)[0]
            for f in feat: F.append([int(user),int(f)])
        np.savetxt(self.outfolder + "/users_attributes.csv", np.array(F), delimiter=",", fmt='%d')

        if self.I.activeItemIndeces:
            p = np.where(self.SalesHistory>=1)
            z = zip(p[0],p[1])
            l = [[i,j] for i,j in z if j in self.I.activeItemIndeces]
            np.savetxt(self.outfolder + "/positive_only_feedback.csv", np.array(l), delimiter=",", fmt='%d')

        if not self.I.activeItemIndeces: self.I.activeItemIndeces = [i for i in range(self.I.totalNumberOfItems)]
        d = []
        for i in self.I.activeItemIndeces:
            feat = np.where(self.I.ItemsFeatures[i]/np.max(self.I.ItemsFeatures[i])>0.33)[0]
            for f in feat: d.append([int(i),int(f)])
        np.savetxt(self.outfolder + "/items_attributes.csv", np.array(d), delimiter=",", fmt='%d')

    def mmlRecommendation(self):
        """ A wrapper around the MyMediaLite toolbox

        Returns:
            recommendations (dict): A {user:[recommended items]} dictionary

        """

        command = "mono MyMediaLite/item_recommendation.exe --training-file=" + self.outfolder + "/positive_only_feedback.csv --item-attributes=" + self.outfolder + "/items_attributes.csv --recommender="+self.algorithm+" --predict-items-number="+str(self.n)+" --prediction-file=" + self.outfolder + "/output.txt --user-attributes=" + self.outfolder + "/users_attributes.csv" # --random-seed="+str(int(self.seed*random.random()))
        os.system(command)

        # Parse output
        f = open( self.outfolder + "/output.txt","r").read()
        f = f.split("\n")
        recommendations = {}
        probabilities = {}

        for line in f[:-1]:
            l = line.split("\t")
            user_id = int(l[0])
            l1 = l[1].replace("[","").replace("]","").split(",")
            rec = [int(i.split(":")[0]) for i in l1]
            prob = [float(i.split(":")[1]) for i in l1]
            probabilities.update(({user_id:prob}))
            recommendations.update({user_id:rec})


        return recommendations, probabilities

class Simulation():

    def __init__(self):
        self.settings = {}

    #TODO change(set) the settings
    def setSettings(self):
        # self.settings = {"Number of active users per day": self.spinBoxUsers.value(),       # Population
        #                  "Days" : self.spinBoxDays.value(),                                 # Number of iterations
        #                  "seed": int(1),
        #                  "Recommender salience": self.spinBoxSalience.value(),
        #                  "Number of published articles per day": self.spinBoxPubArticles.value(),
        #                  "outfolder": "output-"+str(time.time()),
        #                  "Number of recommended articles per day": self.spinBoxRecArticles.value(),
        #                  "Average read articles per day": self.spinBoxUsersArticles.value(),
        #                  "Reading focus": float(self.sliderFocus.value()/100),
        #                  "Recommender algorithms": [str(item.text()) for item in self.comboBoxAlgorithms.selectedItems()],
        #                  "Overall topic weights": [float(i.value()/100) for i in [self.sliderEnt,  self.sliderBus, self.sliderPol, self.sliderSpo, self.sliderTec]],
        #                  "Overall topic prominence": [float(i.value()/10) for i in [self.sliderPromEnt,  self.sliderPromBus, self.sliderPromPol, self.sliderPromSpo, self.sliderPromTec]]}

        self.settings = {"Number of active users per day": 20,       # Population
                         "Days" : 3,                                 # Number of iterations
                         "seed": int(1),
                         "Recommender salience": 5,
                         "Number of published articles per day": 100,
                         "outfolder": "output-"+str(time.time()),
                         "Number of recommended articles per day": 10, #change this to set the desired amount of predicted articles
                         "Average read articles per day": 6,
                         "Reading focus": 0.6,
                         "Recommender algorithms": ['UserAttributeKNN'], # name of the recommender algorithm, can debug the full simulation to get all possible values
                         "Overall topic weights": [float(i/100) for i in [20,  20, 20, 20, 20]], #weights for the topics
                         "Overall topic prominence": [float(i/10) for i in [60,  60, 60, 60, 60]]} # porminence for the topics

        # Make outfolder
        os.makedirs(self.settings["outfolder"])


    #TODO here the sumulation is run need to find a way to make it iteration-esque
    def runSimulation(self):
        """ The main simulation function.

        For different simulation instantiations to run on the same random order of items
        the iterationRange should be the same.

        Args:
            iterationRange (list): The iteration range for the current simulation

        """

        # For all recommenders (starting with the "Control")
        for self.algorithm in self.algorithms:

            # Initialize the iterations range and input data for the recommender
            days = int(self.totalNumberOfIterations/2)
            if self.algorithm == "Control":
                self.iterationRange = [i for i in range(days)]
            else:
                # Copy the users, items and their interactions from the control period
                self.U = copy.deepcopy(ControlU)
                self.I = copy.deepcopy(ControlI)
                self.D = ControlD.copy()  # Start from the control distances between items and users
                self.SalesHistory = ControlHistory.copy()  # Start from the control sale history
                # self.ControlHistory = ControlHistory.copy()  # We use a copy of th


                self.iterationRange = [i for i in range(days,days*2)] #TODO CHANGE ITERATION RANGE TO DESIRED RANGE

            # Start the simulation for the current recommender
            #TODO in this code block the simulation is defined, it runs until the iteration range, need to convert this code
            # block to be in line with what we want...

            for epoch_index, epoch in enumerate(self.iterationRange):

                SalesHistoryBefore = self.SalesHistory.copy()

                self.printj(self.algorithm+": Awareness...")
                self.awarenessModule(epoch)
                InitialAwareness = self.U.Awareness.copy()

                # Recommendation module
                if self.algorithm is not "Control":
                    self.printj(self.algorithm+": Recommendations...")

                    # Call the recommendation object
                    self.Rec.setData(self.U, self.I, self.algorithm, self.SalesHistory)
                    self.Rec.exportToMMLdocuments()
                    recommendations, probabilities = self.Rec.mmlRecommendation()

                    print(probabilities)   #TODO added the probabilty component here

                    # Add recommendations to each user's awareness pool
                    for user in self.U.activeUserIndeces:
                        Rec=np.array([-1])

                        if self.algorithm is not "Control":
                            if user not in recommendations.keys():
                                self.printj(" -- Nothing to recommend -- to user ",user)
                                continue
                            Rec = recommendations[user]
                            self.I.hasBeenRecommended[Rec] = 1
                            self.U.Awareness[user, Rec] = 1

                            # If recommended but previously purchased, minimize the awareness
                            self.U.Awareness[user, np.where(self.SalesHistory[user,Rec]>0)[0] ] = 0

                # Choice
                self.printj(self.algorithm+": Choice...")
                for user in self.U.activeUserIndeces:
                    Rec=np.array([-1])

                    if self.algorithm is not "Control":
                        if user not in recommendations.keys():
                            self.printj(" -- Nothing to recommend -- to user ",user)
                            continue
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
                        self.U.computeNewPositionOfUser(user, self.I.Items[indecesOfChosenItems])

                # Temporal adaptations
                self.printj(self.algorithm+": Temporal adaptations...")
                self.temporalAdaptationsModule(epoch)

                # Compute diversity and other metrics. For this version, we compute
                # only two diversity metrics (EPC, EPD) and the topic distribution
                if self.algorithm is not "Control":

                    self.printj(self.algorithm+": Diversity metrics...")
                    met = metrics.metrics(SalesHistoryBefore, recommendations, self.I.ItemsFeatures, self.I.ItemsDistances, self.SalesHistory)
                    for key in met.keys():
                        self.data["Diversity"][self.algorithm][key].append(met[key])

                    self.printj(self.algorithm+": Distribution...")
                    for i in range(len(self.I.topics)):
                        indeces = np.where(self.I.ItemsClass==i)[0]
                        A = self.SalesHistory[:,indeces] - ControlHistory[:,indeces]
                        self.data["Distribution"][self.algorithm][self.I.topics[i]] = np.sum(np.sum(A,axis=1))

                # Add more metric computations here...

                # Save results
                self.printj(self.algorithm+": Exporting iteration data...")
                self.exportAnalysisDataAfterIteration() #TODO Important piece while here, after each iteration the data metrics are exported


                # After the control period is over, we store its data to be used by the other rec algorithms
                if self.algorithm == "Control":
                    ControlU = copy.deepcopy(self.U)
                    ControlI = copy.deepcopy(self.I)
                    ControlD = self.D.copy()  # Start from the control distances between items and users
                    ControlHistory = self.SalesHistory.copy()  # We use a copy of th


    #TODO used to export the data we want diversity is included
    def exportAnalysisDataAfterIteration(self):
        """ Export data to dataframes

        This is called at the end of each rec algorithm iteration. Certain data
        can be exported for further analysis e.g. the the SalesHistory. For this
        version, we simply export the appropriate data for the figures provided
        by the interface.

        """


        # Metrics output
        df = pd.DataFrame(self.data["Diversity"])
        df.to_pickle(self.outfolder + "/metrics analysis.pkl")

        # Topics distribution output
        df = pd.DataFrame(self.data["Distribution"])
        df.to_pickle(self.outfolder + "/metrics distribution.pkl")


    # initialize the other classes with the settings
    def initWithSettings(self):

        # Simulation inits (not taken from the interface)
        self.AnaylysisInteractionData = []  # Holder for results/data
        self.D = []  # Distance matrix |Users|x|Items| between items and users
        self.SalesHistory = []  # User-item interaction matrix |Users|x|Items|7......7


        # Simulation inits taken from the interface
        self.printj("Initialize simulation class...")
        #sim.gallery = gallery
        self.outfolder = self.settings["outfolder"]
        self.seed = int(self.settings["seed"])
        self.n = int(self.settings["Number of recommended articles per day"])
        self.algorithms = ['Control'] + self.settings["Recommender algorithms"]

        # The totalNumberOfIterations controls the amount of
        # items that will be generated. We first need to run a Control period for
        # iterarionsPerRecommender iterations, on different items than during the
        # recommendation period, as such the total amount of iterations is doubled.
        self.totalNumberOfIterations = int(self.settings["Days"])*2

        self.printj("Initialize users/items classes...")
        U = Users()
        I = Items()

        U.delta = float(self.settings["Recommender salience"])
        U.totalNumberOfUsers = int(self.settings["Number of active users per day"])
        U.seed = int(self.settings["seed"])
        U.Lambda = float(self.settings["Reading focus"])
        U.meanSessionSize = int(self.settings["Average read articles per day"])

        I.seed = int(self.settings["seed"])
        I.topicsFrequency = self.settings["Overall topic weights"]
        I.topicsProminence = self.settings["Overall topic prominence"]
        I.numberOfNewItemsPI = int(self.settings["Number of published articles per day"])

        I.generatePopulation(totalNumberOfIterationsSimulation) #TODO number of iterations of simulation
        U.generatePopulation()

        self.printj("Create simulation instance...")
        self.U = copy.deepcopy(U)
        self.I = copy.deepcopy(I)

        self.D =  euclideanDistance(self.U.Users, self.I.Items)
        self.SalesHistory = np.zeros([self.U.totalNumberOfUsers,self.I.totalNumberOfItems])

        self.printj("Create recommendations instance...")
        self.Rec = Recommendations()
        self.Rec.U = copy.deepcopy(U)
        self.Rec.I = copy.deepcopy(U)
        self.Rec.outfolder = self.settings["outfolder"]
        self.Rec.n = int(self.settings["Number of recommended articles per day"])

        # Create structure to hold data ouput
        self.data = {}
        distribution = {}
        for algorithm in self.algorithms:
            distribution.update({algorithm:{}})
            for key in self.I.topics:
                distribution[algorithm].update({key: 0})
        self.data.update({"Distribution": distribution})

        diversityMetrics = {}  # Holder for diversity metrics (means + std)
        for algorithm in self.algorithms:
            diversityMetrics.update({algorithm:{}})
            for key in ["EPC", "EPCstd",'ILD',"Gini", "EFD", "EPD", "EILD", 'ILDstd', "EFDstd", "EPDstd", "EILDstd"]:
                diversityMetrics[algorithm].update({key:[]})
        self.data.update({"Diversity": diversityMetrics})

    def printj(self, text):
        """template method to replace the printing from the siren simulation until a better solution is found"""
        print(text)

    def awarenessModule(self, epoch):
        """This function computes the awareness of each user.

        While the proximity/prominence awareness is computed in the user class, the current function
        updates that awareness to accommodate for the non-available items and those that the user
        has purchased before. The number of items in the awareness is also limited.

        Args:
        	epoch (int): The current iteration.

        """

        self.U.subsetOfAvailableUsers()
        self.I.subsetOfAvailableItems(epoch)
        self.U.computeAwarenessMatrix(self.D, self.I.ItemProminence, self.I.activeItemIndeces)

        # Adjust for availability
        self.U.Awareness[:,self.I.nonActiveItemIndeces] = 0

        # Adjust for purchase history
        self.U.Awareness = self.U.Awareness - self.SalesHistory>0

        # Adjust for maximum number of items in awareness
        for a in range(self.U.totalNumberOfUsers):
            w = np.where(self.U.Awareness[a,:]==1)[0]
            if len(w)>self.U.w:
                windex = w.tolist()
                random.shuffle(windex)
                self.U.Awareness[a,:] = np.zeros(self.I.totalNumberOfItems)
                self.U.Awareness[a,windex[:self.U.w]] = 1

    def temporalAdaptationsModule(self, epoch):
        """ Update the user-items distances and item- lifespand and prominence.

        Todo:
            * Updating the item-distance only for items that matter

        """

        self.I.updateLifespanAndProminence()

        # We compute this here so that we update the distances between users and not all the items
        self.I.subsetOfAvailableItems(epoch+1)


        if self.algorithm is not "Control":
            D =  euclideanDistance(self.U.Users, self.I.Items[self.I.activeItemIndeces])
            # If you use only a percentage of users then adjust this function
            for u in range(self.U.totalNumberOfUsers): self.D[u,self.I.activeItemIndeces] = D[u,:]

# main function
if __name__ == '__main__':
    sim = Simulation()
    sim.setSettings()
    sim.initWithSettings()
    sim.runSimulation()

    #todo make an exit condition

