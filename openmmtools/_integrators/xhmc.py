import simtk.unit
import simtk.openmm as mm
import numpy as np

from .ghmc import GHMCIntegrator

kB = simtk.unit.BOLTZMANN_CONSTANT_kB * simtk.unit.AVOGADRO_CONSTANT_NA

class XHMCIntegrator(GHMCIntegrator):
    """
    Generalized X hybrid Monte Carlo (XGHMC) integrator.

    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds, steps_per_hmc=10, k_max=2):
        """
        Create an X generalized hybrid Monte Carlo (XGHMC) integrator.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           The collision rate.
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           The integration timestep.
        nsteps : int, optional, default=10
            Number of velocity verlet steps per iteration.

        Notes
        -----
        This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        Metrpolization step to ensure sampling from the appropriate distribution.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        Move initialization of 'sigma' to setting the per-particle variables.

        Examples
        --------

        Create a GHMC integrator.

        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> collision_rate = 91.0 / simtk.unit.picoseconds
        >>> timestep = 1.0 * simtk.unit.femtoseconds
        >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

        References
        ----------
        Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
        http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

        """
        # Create a new custom integrator and initialize all the GHMC variables
        super(XHMCIntegrator, self).__init__(timestep=timestep, collision_rate=collision_rate, steps_per_hmc=steps_per_hmc, temperature=temperature)

        self.k_max = k_max

        self.addGlobalVariable("a", 1.0) # accept or reject

        self.addGlobalVariable("k_max", k_max)  # Maximum number of rounds of dynamics
        self.addGlobalVariable("k", 0)  # Current number of rounds of dynamics
        self.addGlobalVariable("kold", 0)  # Previous value of k stored for debugging purposes
        self.addGlobalVariable("f", 0.0)  # Indicator variable whether this iteration was a flip
        
        self.addGlobalVariable("rho", 0.0)  # temporary variables for acceptance criterion
        self.addGlobalVariable("mu", 0.0)  # 
        self.addGlobalVariable("mu1", 0.0)  # 
        self.addGlobalVariable("mu10", 1.0) # Previous value of mu1
        self.addGlobalVariable("Uold", 0.0)
        self.addGlobalVariable("Unew", 0.0)
        self.addGlobalVariable("uni", 0.0)  # Uniform random variable generated once per round of XHMC

        self.addGlobalVariable("nflip", 0) # number of momentum flips (e.g. complete rejections)
        
        self.add_computations()
        
    def add_computations(self):

        self.addComputeGlobal("s", "step(-k)")  # True only on first step of XHMC round
        self.addComputeGlobal("l", "step(k - k_max)")  # True only only last step of XHMC round
        #
        # Allow context updating here.
        #
        self.addUpdateContextState()

        #
        # Constrain positions.
        #
        self.addConstrainPositions()

        #
        # Velocity perturbation.
        self.addComputePerDof("v", "s * (sqrt(b) * v + (1 - s) * sqrt(1 - b) * sigma * gaussian) + (1 - s) * v")
        self.addConstrainVelocities()

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "s * (ke + energy) + (1 - s) * Eold")
        self.addComputePerDof("xold", "s * x + (1 - s) * xold")
        self.addComputePerDof("vold", "s * v + (1 - s) * vold")
        
        self.addComputeGlobal("mu1", "mu1 * (1 - s)")
        self.addComputeGlobal("uni", "(1 - s) * uni + uniform * s")

        self.add_hmc_loop()

        self.addComputeSum("ke", "0.5*m*v*v")

        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("Unew", "energy")
        self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
        self.addComputeGlobal("mu", "min(1, r)")
        self.addComputeGlobal("mu1", "max(mu1, mu)")
        self.addComputeGlobal("a", "step(mu1 - uni)")
        self.addComputePerDof("x", "x * a + xold * (1 - a)")
        self.addComputePerDof("v", "v * a - vold * (1 - a)")
        
        self.addComputeGlobal("f", "(1 - a) * l")  # Flip is True ONLY on rejection at last cycle
        
        self.addComputeGlobal("kold", "k")  # Store the previous value of k for debugging purposes
        self.addComputeGlobal("k", "(k + 1) * (1 - f) * (1 - a)")  # Increment by one ONLY if not flipping momenta or accepting, otherwise set to zero

        # Accumulate statistics.
        #
        self.addComputeGlobal("nflip", "nflip + f")
        self.addComputeGlobal("naccept", "naccept + a")
        self.addComputeGlobal("ntrials", "ntrials + 1")

    @property
    def n_accept(self):
        """The number of accepted HMC moves."""
        return self.getGlobalVariableByName("naccept")

    @property
    def n_trials(self):
        """The total number of attempted HMC moves."""
        return self.getGlobalVariableByName("ntrials")

    @property
    def n_flip(self):
        """The total number of momentum flips."""
        return self.getGlobalVariableByName("nflip")

    @property
    def acceptance_rate(self):
        """The acceptance rate:"""
        return 1.0 - (self.k_max + 1) * self.n_flip / float(self.n_trials)
