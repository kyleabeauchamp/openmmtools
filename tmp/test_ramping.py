import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

collision_rate = 1.0 / u.picoseconds
n_steps = 5000
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water


def test_hmc(timestep, steps_per_hmc, alpha):
    timestep = timestep * u.femtoseconds
    integrator = integrators.RampedHMCIntegrator(temperature, steps_per_hmc, timestep, max_boost=alpha)
    context = mm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    integrator.step(n_steps)    
    return integrator.acceptance_rate

timestep_list = np.array([2.25, 2.3])
alpha_list = np.array([0.15, 0.25])
steps_per_hmc = 50
data = []
for i, timestep in enumerate(timestep_list):
    for j, alpha in enumerate(alpha_list):
        print(i, j, timestep, alpha)
        acceptance = test_hmc(timestep, steps_per_hmc, alpha)
        data.append(dict(acceptance=acceptance, timestep=timestep, alpha=alpha))
        print(data[-1])
        
data = pd.DataFrame(data)
data = data.pivot("timestep", "alpha", "acceptance")
tdata = (data.T * data.index.values).T

