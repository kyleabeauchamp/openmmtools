import numpy as np
import statsmodels.api as sm
import pandas as pd
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit as u
import openmmtools

cas = "tip3p"

def test_hmc(timestep, steps_per_hmc):
    output_frequency = 1
    n_steps = 2000
    timestep = timestep * u.femtoseconds
    temperature = 300. * u.kelvin
    pressure = 1.0 * u.atmospheres
    barostat_frequency = 1
    cutoff = 1.0 * u.nanometers
    ffxml_filename = "%s.xml" % cas
    ff = app.ForceField(ffxml_filename)
    log_filename = "./production.log"
    pdb = app.PDBFile("./tip3p.pdb")
    topology = pdb.topology
    positions = pdb.positions
    system = ff.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=app.HBonds)
    integrator = openmmtools.integrators.HMCIntegrator(temperature, steps_per_hmc, timestep)
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))
    
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(app.StateDataReporter(open(log_filename, 'w'), output_frequency, step=True, time=True, speed=True, density=True))
    simulation.step(n_steps)
    return (timestep / u.femtoseconds, steps_per_hmc, integrator.n_accept, integrator.n_trials, integrator.acceptance_rate)

timestep_list = np.array([0.1, 0.5, 0.75, 1.0, 1.25, 1.5])
steps_per_hmc_list = np.array([5, 10, 25, 50, 100])
data = []
for i, timestep in enumerate(timestep_list):
    for j, steps_per_hmc in enumerate(steps_per_hmc_list):
        acceptance = test_hmc(timestep, steps_per_hmc)[-1]
        data.append(dict(acceptance=acceptance, timestep=timestep, steps_per_hmc=steps_per_hmc))
        
data = pd.DataFrame(Data)
data = data.pivot("timestep", "steps_per_hmc", "acceptance")
