from baresimulation import Simulation
import pandas as pd

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

def experiment_a():
    # a_arr = [1 + (i*.2) for i in range(16)]
    print("START EXPERIMENT")
    a_arr = [1.8, 2, 2.2]
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
    pass

def experiment_c():
    pass

def experiment_pi():
    pass

def experiment_particles():
    pass

def experiment_generations():
    pass

def experiment_users():
    pass

def experiment_articles():
    pass


experiment_a()