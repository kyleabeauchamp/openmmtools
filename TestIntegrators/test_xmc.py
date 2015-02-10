import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

steps_per_hmc = 5
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()
timestep = 0.3 * u.femtoseconds
n_steps = 50

integrator = integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(n_steps)
r1 = integrator.acceptance_rate

integrator0 = integrators.XHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, k_max=1)
context = mm.Context(testsystem.system, integrator0)
context.setPositions(testsystem.positions)f
context.setVelocitiesToTemperature(temperature)
integrator0.step(n_steps)
r0 = integrator0.acceptance_rate


integrator = integrator0
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
