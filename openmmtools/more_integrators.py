import numpy as np

import simtk.unit

import simtk.unit as units
import simtk.openmm as mm

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# INTEGRATORS
#=============================================================================================


class GHMC2(mm.CustomIntegrator):
    """Generalized hybrid Monte Carlo (GHMC) integrator.
    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, steps_per_hmc=10, timestep=1*simtk.unit.femtoseconds, collision_rate=91.0/simtk.unit.picoseconds):
        """Create a generalized hybrid Monte Carlo (GHMC) integrator.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        steps_per_hmc : int, default: 10
           The number of velocity Verlet steps to take per round of hamiltonian dynamics
           This must be an even number!
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.  The total time taken per iteration
           will equal timestep * steps_per_hmc
        collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: 91 / picoseconds
           The collision rate for the langevin velocity corruption.
        """

        mm.CustomIntegrator.__init__(self, timestep)
        
        self.steps_per_hmc = steps_per_hmc

        # Compute the thermal energy.
        self.temperature = temperature
        self.collision_rate = collision_rate        
        self.timestep = timestep
        self.create()

    @property
    def kT(self):
        """The thermal energy."""
        return kB * self.temperature

    @property
    def b(self):
        """The scaling factor for preserving versus randomizing velocities."""
        return np.exp(-self.collision_rate * self.timestep)

    def create(self):
        self.initialize_variables()
        self.add_draw_velocities_step()
        self.add_cache_variables_step()
        self.add_hmc_iterations()
        self.add_accept_or_reject_step()
        self.add_accumulate_statistics_step()


    def initialize_variables(self):
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addPerDofVariable("vold", 0) # old velocities
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addPerDofVariable("x1", 0) # for constraints
        self.addGlobalVariable("b", self.b) # velocity mixing parameter                

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities();

    def add_cache_variables_step(self):
        """Store old positions and energies."""
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")


    def add_accumulate_statistics_step(self):
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")


    def add_accept_or_reject_step(self):
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "x*accept + xold*(1-accept)")
        self.addComputePerDof("v", "v*accept - vold*(1-accept)")

    def build_timestep_ramp(self):
        """Construct a linearly ramped grid of timesteps that satisfies detailed balance."""
        raw_grid = lambda n: np.array(range(n / 2) + range(n / 2)[::-1])
        rho_func = lambda n: 1 - self.max_boost + raw_grid(n) * 2 * self.max_boost / (n / 2 - 1.)
        
        self.rho_grid = rho_func(self.steps_per_hmc)
        
        print(self.steps_per_hmc, self.rho_grid.sum())

    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics, with ramping step sizes."""
        print("Adding GHMC2 steps.")
        for step in range(self.steps_per_hmc):
            self.addComputePerDof("v", "v+0.5*dt*f/m")
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
            self.addConstrainVelocities()

    @property
    def n_accept(self):
        """The number of accepted HMC moves."""
        return self.getGlobalVariableByName("naccept")

    @property
    def n_trials(self):
        """The total number of attempted HMC moves."""
        return self.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        """The acceptance rate: n_accept  / n_trials."""
        return self.n_accept / float(self.n_trials)


class RampedHMCIntegrator(GHMC2):
    """Hybrid Monte Carlo (HMC) integrator with linearly ramped non-uniform timesteps.
    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, steps_per_hmc=10, timestep=1*simtk.unit.femtoseconds, collision_rate=91.0/simtk.unit.picoseconds, max_boost=0.0):
        """Create a hybrid Monte Carlo (HMC) integrator with linearly ramped non-uniform timesteps.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        steps_per_hmc : int, default: 10
           The number of velocity Verlet steps to take per round of hamiltonian dynamics
           This must be an even number!
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.  The total time taken per iteration
           will equal timestep * steps_per_hmc
        collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: 91 / picoseconds
           The collision rate for the langevin velocity corruption.
        max_boost : float, default=0.0
            Control parameter for linearly ramping of timestep.
        """

        mm.CustomIntegrator.__init__(self, timestep)

        if steps_per_hmc % 2 != 0:
            raise(ValueError("steps_per_hmc must be an even number!"))
        
        self.steps_per_hmc = steps_per_hmc

        self.collision_rate = collision_rate
        self.timestep = timestep
        self.temperature = temperature

        self.max_boost = max_boost
        self.build_timestep_ramp()
        
        self.create()

    def build_timestep_ramp(self):
        """Construct a linearly ramped grid of timesteps that satisfies detailed balance."""
        raw_grid = lambda n: np.array(range(n / 2) + range(n / 2)[::-1])
        rho_func = lambda n: 1 - self.max_boost + raw_grid(n) * 2 * self.max_boost / (n / 2 - 1.)
        
        self.rho_grid = rho_func(self.steps_per_hmc)
        
        print(self.steps_per_hmc, self.rho_grid.sum())

    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics, with ramping step sizes."""
        print("Adding ramped HMC steps.")
        for step in range(self.steps_per_hmc):
            rho = self.rho_grid[step]
            self.addComputePerDof("v", "v+%f*0.5*dt*f/m" % rho)
            self.addComputePerDof("x", "x+%f*dt*v" % rho)
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+%f*0.5*dt*f/m+(x-x1)/dt/%f" % (rho, rho))
            self.addConstrainVelocities()

