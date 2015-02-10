import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 5
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()

timestep = 0.20 * u.femtoseconds
integrator = integrators.XHMCIntegrator(temperature=temperature, nsteps=n_steps, timestep=timestep, k_max=4)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)

for i in range(25):
    integrator.step(1)
    f = lambda key: integrator.getGlobalVariableByName(key)
    print("AFTER ITERATION********")
    print("kold=%d k=%d uni=%f" % ((f("kold"), f("k"), f("uni"))))
    print("Enew=%f Eold=%f Enew-Eold=%f" % (f("Enew"), f("Eold"), f("Enew")-f("Eold")))
    print("rho=%f, mu=%f, mu1old=%f mu1=%f, delta=%f flip=%d, accept=%d"% (f("rho"), f("mu"), f("mu10"), f("mu1"), f("mu1") - f("uni"), f("f"), f("a")))
    state = context.getState(getPositions=True, getEnergy=True)
    xyz = state.getPositions(asNumpy=True)
    energy = state.getPotentialEnergy() + state.getKineticEnergy()

integrator.acceptance_rate
