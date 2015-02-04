import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 25
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()

timestep = 1.0 * u.femtoseconds
integrator = integrators.XHMCIntegrator(temperature=temperature, nsteps=n_steps, timestep=timestep)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
integrator.step(n_steps)
state = context.getState(getPositions=True)
xyz = state.getPositions(asNumpy=True)
