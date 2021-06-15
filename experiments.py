from baresimulation import Simulation
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None, "display.max_columns", None)

a = 2
b = 2
c = 2
num_particles = 8
num_generations = 20
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
    met_arr = pd.DataFrame()

    for a in a_arr:
        sim = Simulation()
        sim.setSettings()
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
    pi_arr = [[1.8, 1.2, 1.2, 1.2, 1.2, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [2, 1.1, 1.1, 1.1, 1.1, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]]
    print("START EXPERIMENT")
    met_arr = pd.DataFrame()
    day_arr = np.array([])
    exp_arr = []
    for ii, pi in enumerate(pi_arr):
        print("Testing pi = ", pi)
        sim = Simulation()
        sim.setSettings()
        sim.initWithSettings()
        sim.runSimulation(a, b, c, pi, num_particles, num_generations)
        met_arr = met_arr.append(sim.met_out, ignore_index=True)
        day_arr = np.append(day_arr, np.arange(1, 10))
        exp_arr = exp_arr + ([ii + 1] * 9)

    print(day_arr)
    print(exp_arr)
    met_arr.insert(0, 'day', day_arr)
    met_arr.insert(0, 'exp num', exp_arr)
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