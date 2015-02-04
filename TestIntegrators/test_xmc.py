import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 5
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()

timestep = 0.1 * u.femtoseconds
integrator = integrators.XHMCIntegrator(temperature=temperature, nsteps=n_steps, timestep=timestep, k_max=100)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)

for i in range(10):
    integrator.step(1)
    print(integrator.n_trials, integrator.n_accept, integrator.n_flip, integrator.acceptance_rate)
    f = lambda key: integrator.getGlobalVariableByName(key)
    print("rho=%f, mu=%f, mu1=%f, flip=%f" % (f("rho"), f("mu"), f("mu1"), f("flip")))
    state = context.getState(getPositions=True, getEnergy=True)
    xyz = state.getPositions(asNumpy=True)
    energy = state.getPotentialEnergy()
    print(energy)
    print(xyz)
