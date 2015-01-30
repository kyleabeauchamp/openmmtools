import pandas as pd
import time
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems


groups = [(0, 4), (1, 2), (2, 1)]

n_steps = 1000
temperature = 300. * u.kelvin

#testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
testsystem = testsystems.MethanolBox()
integrators.guess_force_groups(testsystem.system)

def test_hmc(timestep, steps_per_hmc, alpha):
    timestep = timestep * u.femtoseconds
    integrator = integrators.HMCRespaIntegrator(groups, temperature, steps_per_hmc, timestep)
    context = mm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    integrator.step(1)
    t0 = time.time()
    integrator.step(n_steps)    
    t1 = time.time()
    elapsed = t1 - t0
    performance = (timestep / u.femtoseconds) * integrator.acceptance_rate / elapsed
    base_performance = (timestep / u.femtoseconds) * integrator.acceptance_rate 
    return performance, base_performance, elapsed, integrator.acceptance_rate

timestep_list = np.array([6.0, 7.0, 8.0, 9.0])
alpha_list = np.array([0.0])
steps_per_hmc = 50
data = []
for i, timestep in enumerate(timestep_list):
    for j, alpha in enumerate(alpha_list):
        print(i, j, timestep, alpha)
        performance, base_performance, elapsed, acceptance = test_hmc(timestep, steps_per_hmc, alpha)
        data.append(dict(performance=performance, base_performance=base_performance, elapsed=elapsed, acceptance=acceptance, timestep=timestep, alpha=alpha))
        print(data[-1])
        
data = pd.DataFrame(data)
data


"""
Water box:
   acceptance  alpha  base_performance     elapsed  performance  timestep
0    0.623377      0          3.740260  121.191959     0.030862         6
1    0.613387      0          4.293706  124.812997     0.034401         7
2    0.515485      0          4.123876  127.224771     0.032414         8
3    0.000000      0          0.000000  235.574955     0.000000         9
"""
