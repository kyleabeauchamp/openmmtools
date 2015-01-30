import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 5000
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
#testsystem = testsystems.SrcExplicit()

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

"""
HOH 20 steps

In [18]: data
Out[18]: 
alpha       0.00    0.01    0.02    0.03    0.04    0.05    0.06
timestep                                                        
1.9       0.6776  0.6716  0.6976  0.6974  0.6906  0.7076  0.6928
2.0       0.6686  0.6880  0.6760  0.6984  0.6978  0.0000  0.6890
2.1       0.6372  0.6586  0.6640  0.6758  0.6710  0.6948  0.0000

In [19]: tdata
Out[19]: 
alpha        0.00     0.01     0.02     0.03     0.04     0.05     0.06
timestep                                                               
1.9       1.28744  1.27604  1.32544  1.32506  1.31214  1.34444  1.31632
2.0       1.33720  1.37600  1.35200  1.39680  1.39560  0.00000  1.37800
2.1       1.33812  1.38306  1.39440  1.41918  1.40910  1.45908  0.00000

"""
