import time
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 100
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()

steps_per_hmc = 10
max_boost = 0.01

timestep = 6.0 * u.femtoseconds
groups = [(0, 4), (1, 2), (2, 1)]

integrators.guess_force_groups(testsystem.system)

integrator = integrators.RampedHMCRespaIntegrator(groups, temperature, steps_per_hmc, timestep, max_boost=max_boost)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)

integrator.step(1)
t0 = time.time()
integrator.step(n_steps)    
t1 = time.time()
dt = t1 - t0

performance = timestep * integrator.acceptance_rate / dt
