import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

collision_rate = 1.0 / u.picoseconds
n_steps = 500
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water


def test_hmc(timestep, steps_per_hmc, alpha):
    timestep = timestep * u.femtoseconds
    integrator = integrators.RampedHMCIntegrator(temperature, steps_per_hmc, timestep, max_boost=alpha)
    context = mm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    integrator.step(n_steps)    
    return integrator.acceptance_rate

#timestep_list = np.array([1.0, 1.5])
timestep_list = np.linspace(1.5, 2.5, 5)
alpha_list = np.array([0.0, 0.1, 0.2])
steps_per_hmc = 26
data = []
for i, timestep in enumerate(timestep_list):
    for j, alpha in enumerate(alpha_list):
        print(i, j, timestep, alpha)
        acceptance = test_hmc(timestep, steps_per_hmc, alpha)
        data.append(dict(acceptance=acceptance, timestep=timestep, alpha=alpha, normalized=timestep * acceptance))
        print(data[-1])
        
data = pd.DataFrame(data)
acceptance = data.pivot("timestep", "alpha", "acceptance")
normalized = data.pivot("timestep", "alpha", "normalized")

acceptance
normalized
