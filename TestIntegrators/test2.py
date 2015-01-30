import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

#n_steps = 1500
n_steps = 500
temperature = 300. * u.kelvin
#testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
testsystem = testsystems.SrcExplicit()

def test_hmc(timestep, steps_per_hmc):
    timestep = timestep * u.femtoseconds
    integrator = integrators.RampedHMCIntegrator(temperature, steps_per_hmc, timestep, 0.1)
    context = mm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    integrator.step(n_steps)    
    return integrator.acceptance_rate

#timestep_list = np.array([0.5, 1.0, 1.5, 1.9, 2.0, 2.1])
timestep_list = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
steps_per_hmc_list = np.array([2, 3, 4, 5, 10])
data = []
for i, timestep in enumerate(timestep_list):
    for j, steps_per_hmc in enumerate(steps_per_hmc_list):
        print(i, j, timestep, steps_per_hmc)
        acceptance = test_hmc(timestep, steps_per_hmc)
        data.append(dict(acceptance=acceptance, timestep=timestep, steps_per_hmc=steps_per_hmc))
        
data = pd.DataFrame(data)
data = data.pivot("timestep", "steps_per_hmc", "acceptance")
tdata = (data.T * data.index.values).T

"""
3.18 nm water box:

In [17]: data
Out[17]: 
steps_per_hmc       10        25        50        100
timestep                                             
0.5            0.863333  0.783333  0.754667  0.692667
1.0            0.804000  0.748667  0.688667  0.659333
1.5            0.763333  0.714667  0.657333  0.601333
1.9            0.735333  0.656667  0.632667  0.593333
2.0            0.693333  0.612000  0.622000  0.561333
2.1            0.712000  0.631333  0.608000  0.614000

In [18]: tdata
Out[18]: 
steps_per_hmc       10        25        50        100
timestep                                             
0.5            0.431667  0.391667  0.377333  0.346333
1.0            0.804000  0.748667  0.688667  0.659333
1.5            1.145000  1.072000  0.986000  0.902000
1.9            1.397133  1.247667  1.202067  1.127333
2.0            1.386667  1.224000  1.244000  1.122667
2.1            1.495200  1.325800  1.276800  1.289400

"""
