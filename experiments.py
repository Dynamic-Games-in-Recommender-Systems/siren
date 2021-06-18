from baresimulation import Simulation
import pandas as pd
import numpy as np
import time
pd.set_option("display.max_rows", None, "display.max_columns", None)

a = 2
b = 2
c = 2
num_particles = 16
num_generations = 200
# pi = [
#     1.8,
#     1.2,
#     1.2,
#     1.2,
#     1.2,
#     1.05,
#     1.05,
#     1.05,
#     1.05,
#     1.05,
#     1.05,
#     1.05,
#     1.05
# ]

def experiment_a():
    # a_arr = [1 + (i*.2) for i in range(16)]
    print("START EXPERIMENT")
    a_arr = [1, 2, 3]
    users_number = [5, 10, 20]
    met_arr = pd.DataFrame()

    for a in a_arr:
        sim = Simulation()
        sim.setSettings(users_number[a])
        sim.initWithSettings()
        sim.runSimulation(a, b, c, pi, num_particles, num_generations)
        met_arr = met_arr.append(sim.met, ignore_index=True)

    print(met_arr)

    ## Return the metrics and maybe store in csv file.

def experiment_b():
    b_arr = [i*0.4 for i in range(21)]
    print("START EXPERIMENT")
    b_arr = [0, 1, 2]
    met_arr = pd.DataFrame()

    for b in b_arr:
        sim = Simulation()
        sim.setSettings()
        sim.initWithSettings()
        sim.runSimulation(a, b, c, pi, num_particles, num_generations)
        met_arr = met_arr.append(sim.met_out, ignore_index=True)

    print(met_arr)

def experiment_c():
    pass

def experiment_pi():
    # pi_arr = [[1.8, 1.2, 1.2, 1.2, 1.2, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05],
    #           [2, 1.1, 1.1, 1.1, 1.1, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01],
    #           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    pi_arr = [[1.8, 1.2, 1.2, 1.2, 1.2, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05]]
    # pi_arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
    users_number = [5, 10, 20]
    print("START EXPERIMENT")
    met_arr = pd.DataFrame()
    day_arr = np.array([])
    exp_arr = []
    time_arr = []
    first_run = True

    for ii, pi in enumerate(pi_arr):
        for user in users_number:

            print("Testing pi = ", pi)

            # Run base case
            if first_run:
                first_run = False
                sim = Simulation()
                sim.setSettings(user)
                sim.initWithSettings()
                sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=False)

                met_arr = met_arr.append(sim.met_out, ignore_index=True)
                time_arr = time_arr + sim.time
                day_arr = np.append(day_arr, np.arange(1, len(time_arr) + 1))
                exp_arr = exp_arr + (['Base'] * len(time_arr))

            days = len(time_arr)

            sim = Simulation()
            sim.setSettings(user)
            sim.initWithSettings()
            sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=True)

            met_arr = met_arr.append(sim.met_out, ignore_index=True)
            time_arr = time_arr + sim.time
            day_arr = np.append(day_arr, np.arange(1, days + 1)) #TODO this is hardcoded
            exp_arr = exp_arr + (['Pi' + str(ii + 1)] * days)

    #
    # print(len(time_arr))
    # print(len(day))
    met_arr['Time'] = time_arr
    met_arr['Day'] = day_arr
    met_arr['Exp'] = exp_arr

    met_arr.to_csv('Results\pi-experiment-' + str(time.time())+ ".csv")
    met_arr.to_excel('Results\pi-experiment-' + str(time.time())+ '.xlsx')
    print(met_arr)

def experiment_particles():
    pass

def experiment_generations():
    pass

def experiment_users():
    pass

def experiment_articles():
    pass


experiment_pi()
