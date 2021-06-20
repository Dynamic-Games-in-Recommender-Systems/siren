from baresimulation import Simulation
import pandas as pd
import numpy as np
import time
pd.set_option("display.max_rows", None, "display.max_columns", None)
from tqdm import tqdm

a = 2
b = 2
c = 2
num_particles = 16
num_generations = 200
pi = [
    1.8,
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
    1.05
]

def experiment_pi():
    pi_arr = [[1.8, 1.2, 1.2, 1.2, 1.2, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05],
              [2, 1.1, 1.1, 1.1, 1.1, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # pi_arr = [[1.8, 1.2, 1.2, 1.2, 1.2, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05]]
    # pi_arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
    print("START EXPERIMENT")
    met_arr = pd.DataFrame()
    day_arr = np.array([])
    exp_arr = []
    time_arr = []


    for _ in tqdm(range(3)):
        first_run = True

        for ii, pi in enumerate(pi_arr):
            print("Testing pi = ", pi)

            # Run base case
            if first_run:
                first_run = False
                sim = Simulation()
                sim.setSettings()
                sim.initWithSettings()
                sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=False)

                met_arr = met_arr.append(sim.met_out, ignore_index=True)
                time_arr = time_arr + sim.time
                day_arr = np.append(day_arr, np.arange(1, 11))
                exp_arr = exp_arr + (['Base'] * 10)


            sim = Simulation()
            sim.setSettings()
            sim.initWithSettings()
            sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=True)

            met_arr = met_arr.append(sim.met_out, ignore_index=True)
            time_arr = time_arr + sim.time
            day_arr = np.append(day_arr, np.arange(1, 11))
            exp_arr = exp_arr + (['Pi' + str(ii + 1)] * 10)

    #
    # print(len(time_arr))
    # print(len(day))
    met_arr['Time'] = time_arr
    met_arr['Day'] = day_arr
    met_arr['Exp'] = exp_arr

    met_arr.to_csv('Results\pi-experiment-' + str(time.time())+ ".csv")
    met_arr.to_excel('Results\pi-experiment-' + str(time.time())+ '.xlsx')
    print(met_arr)



experiment_pi()
