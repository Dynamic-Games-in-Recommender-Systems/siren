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
num_generations = 500
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
    print("START EXPERIMENT")
    met_arr = pd.DataFrame()
    day_arr = np.array([])
    exp_arr = []
    time_arr = []

    for _ in tqdm(range(10)):
        first_run = True

        # Run base case
        if first_run:
            first_run = False
            sim = Simulation()
            sim.setSettings()
            sim.initWithSettings()
            sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=False)

            met_arr = met_arr.append(sim.met_out, ignore_index=True)
            time_arr = time_arr + sim.time
            day_arr = np.append(day_arr, np.arange(1, 21))
            exp_arr = exp_arr + (['Base'] * 20)


        sim = Simulation()
        sim.setSettings()
        sim.initWithSettings()
        sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=True)

        met_arr = met_arr.append(sim.met_out, ignore_index=True)
        time_arr = time_arr + sim.time
        day_arr = np.append(day_arr, np.arange(1, 21))
        exp_arr = exp_arr + (['Opt'] * 20)

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
