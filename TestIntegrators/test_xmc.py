import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

k_max = 0
steps_per_hmc = 5
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()
timestep = 0.3 * u.femtoseconds
n_steps = 1000


integrator = integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(100)
positions = context.getState(getPositions=True).getPositions()

integrator = integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
context = mm.Context(testsystem.system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(n_steps)
r1 = integrator.acceptance_rate

integrator0 = integrators.XHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, k_max=k_max)
context = mm.Context(testsystem.system, integrator0)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)
integrator0.step(n_steps)
r0 = integrator0.acceptance_rate
