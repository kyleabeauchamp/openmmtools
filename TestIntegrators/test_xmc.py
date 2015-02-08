import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 5
temperature = 300. * u.kelvin
testsystem = testsystems.WaterBox()

timestep = 0.25 * u.femtoseconds
integrator = integrators.XHMCIntegrator(temperature=temperature, nsteps=n_steps, timestep=timestep, k_max=4)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)

for i in range(10):
    integrator.step(1)
    f = lambda key: integrator.getGlobalVariableByName(key)
    print("*************")
    print("k=%d uni=%f" % (f("k"), f("uni")))
    print("Uold=%f Unew=%f Enew=%f Eold=%f Enew-Eold=%f" % (f("Uold"), f("Unew"), f("Enew"), f("Eold"), f("Enew")-f("Eold")))
    print("rho=%f, mu=%f, mu1old=%f mu1=%f, flip=%d, accept=%d"% (f("rho"), f("mu"), f("mu10"), f("mu1"), f("flip"), f("accept")))
    state = context.getState(getPositions=True, getEnergy=True)
    xyz = state.getPositions(asNumpy=True)
    energy = state.getPotentialEnergy()
