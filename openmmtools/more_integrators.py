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


class RampedHMCIntegrator(mm.CustomIntegrator):
    """
    Hybrid Monte Carlo (HMC) integrator.

    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, nsteps=10, timestep=1*simtk.unit.femtoseconds, collision_rate=91.0/simtk.unit.picoseconds, max_boost=0.1):
        """
        Create a hybrid Monte Carlo (HMC) integrator.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        nsteps : int, default: 10
           The number of velocity Verlet steps to take per HMC trial.
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.

        Warning
        -------
        Because 'nsteps' sets the number of steps taken, a call to integrator.step(1) actually takes 'nsteps' steps.

        Notes
        -----
        The velocity is drawn from a Maxwell-Boltzmann distribution, then 'nsteps' steps are taken,
        and the new configuration is either accepted or rejected.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        Currently, the simulation timestep is only advanced by 'timestep' each step, rather than timestep*nsteps.  Fix this.

        Examples
        --------

        Create an HMC integrator.

        >>> timestep = 1.0 * simtk.unit.femtoseconds # fictitious timestep
        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> nsteps = 10 # number of steps per call
        >>> integrator = HMCIntegrator(temperature, nsteps, timestep)

        """

        super(RampedHMCIntegrator, self).__init__(timestep)
        
        self.steps_per_hmc = nsteps

        # Compute the thermal energy.
        kT = kB * temperature

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addPerDofVariable("x1", 0) # for constraints
        self.addGlobalVariable("b", np.exp(-collision_rate * timestep)) # velocity mixing parameter        

        #
        # Pre-computation.
        # This only needs to be done once, but it needs to be done for each degree of freedom.
        # Could move this to initialization?
        #
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Allow Context updating here, outside of inner loop only.
        #
        self.addUpdateContextState();

        #
        # Draw new velocity.
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities();

        #
        # Store old position and energy.
        #
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")

        #
        # Inner symplectic steps using velocity Verlet.
        #
        raw_grid = lambda n: np.array(range(n / 2) + range(n / 2)[::-1])
        rho_func = lambda n: 1 - max_boost + raw_grid(n) * 2 * max_boost / (n / 2 - 1.)
        rho_grid = rho_func(nsteps)
        print(nsteps, rho_grid.sum())            

        self.add_hmc_iterations(rho_grid)

        #
        # Accept/reject step.
        #
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "x*accept + xold*(1-accept)")

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")


    def add_hmc_iterations(self, rho_grid):
        """Add self.n_steps iterations of hamiltonian dynamics, with ramping step sizes."""
        for step in range(self.steps_per_hmc):
            rho = rho_grid[step]
            self.addComputePerDof("v", "v+%f*0.5*dt*f/m" % rho)
            self.addComputePerDof("x", "x+%f*dt*v" % rho)
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+%f*0.5*dt*f/m+(x-x1)/dt/%f" % (rho, rho))
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
