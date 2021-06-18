from baresimulation import Simulation
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None, "display.max_columns", None)

a = 2
b = 2
c = 2
num_particles = 8
num_generations = 20
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
    # pi_arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
    print("START EXPERIMENT")
    met_arr = pd.DataFrame()
    day_arr = np.array([])
    exp_arr = []
    time_arr = []
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
            exp_arr = exp_arr + (['Base'] * 10)

        sim = Simulation()
        sim.setSettings()
        sim.initWithSettings()
        sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=True)
        met_arr = met_arr.append(sim.met_out, ignore_index=True)
        day_arr = np.append(day_arr, np.arange(1, 11))
        exp_arr = exp_arr + (['Pi' + str(ii + 1)] * 10)

    met_arr['Time'] = time_arr
    met_arr['Day'] = day_arr
    met_arr['Exp'] = exp_arr
    print(met_arr)


def experiment_particles():
    particle_arr = [2, 10, 20]
    print("START EXPERIMENT")
    met_arr = pd.DataFrame()
    day_arr = np.array([])
    exp_arr = []
    time_arr = []
    first_run = True

    for ii, num_particles in enumerate(particle_arr):
        print("Testing num_particles = ", num_particles)

        # Run base case
        if first_run:
            first_run = False
            sim = Simulation()
            sim.setSettings()
            sim.initWithSettings()
            sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=False)
            met_arr = met_arr.append(sim.met_out, ignore_index=True)
            time_arr = time_arr + sim.time
            exp_arr = exp_arr + (['Base'] * 10)

        sim = Simulation()
        sim.setSettings()
        sim.initWithSettings()
        sim.runSimulation(a, b, c, pi, num_particles, num_generations, game_trigger=True)
        met_arr = met_arr.append(sim.met_out, ignore_index=True)
        day_arr = np.append(day_arr, np.arange(1, 11))
        exp_arr = exp_arr + ([str(num_particles)] * 10)

    met_arr['Time'] = time_arr
    met_arr['Day'] = day_arr
    met_arr['Exp'] = exp_arr
    print(met_arr)
    met_arr.to_csv('PI experiments.csv')

# This experiment is different <- needs all EPC values for each generation for each day
def experiment_generations():
    num_generations = 200

def experiment_users():
    use_arr = [5, 10, 20]

def experiment_articles():
    articles_arr = [5, 13, 20]

