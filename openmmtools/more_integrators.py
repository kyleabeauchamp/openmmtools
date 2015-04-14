import time
import pandas as pd
import numpy as np

import simtk.unit
u = simtk.unit

import simtk.unit as units
import simtk.openmm as mm

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# INTEGRATORS
#=============================================================================================

def guess_force_groups(system, separate_reciprical=True):
    """Set NB short-range to 1 and long-range to 2, which is usually OK."""
    for force in system.getForces():
        if isinstance(force, mm.openmm.NonbondedForce):
            force.setForceGroup(1)
            if separate_reciprical:
                force.setReciprocalSpaceForceGroup(2)
            else:
                force.setReciprocalSpaceForceGroup(1)


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
        self.temperature = temperature
        self.collision_rate = collision_rate        
        self.timestep = timestep
        self.create()
    
    def step(self, n_steps):
        if not hasattr(self, "elapsed_time"):
            self.elapsed_time = 0.0
        if not hasattr(self, "elapsed_steps"):
            self.elapsed_steps = 0.0
        
        t0 = time.time()
        mm.CustomIntegrator.step(self, n_steps)        
        self.elapsed_time += time.time() - t0
        
        self.elapsed_steps += self.steps_per_hmc

    @property
    def time_per_step(self):
        return (self.elapsed_time / self.elapsed_steps)

    @property
    def days_per_step(self):
        return self.time_per_step / (60. * 60. * 24.)

    @property
    def effective_ns_per_day(self):
        return (self.effective_timestep / self.days_per_step) / u.nanoseconds

    @property
    def ns_per_day(self):
        return (self.timestep / self.days_per_step) / u.nanoseconds

    def vstep(self, n_steps, verbose=False):
        """Do n_steps of dynamics and return a summary dataframe."""

        data = []
        for i in range(n_steps):
            
            self.step(1)

            d = self.summary()
            data.append(d)
        data = pd.DataFrame(data)
        
        data["effective_ns_per_day"] = self.effective_ns_per_day
        data["ns_per_day"] = self.ns_per_day
        data["time_per_step"] = (self.elapsed_time / self.elapsed_steps)
        print(data)
        return data

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
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")        

    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics."""
        print("Adding GHMC2 steps.")
        for step in range(self.steps_per_hmc):
            self.addComputePerDof("v", "v+0.5*dt*f/m")
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
            self.addConstrainVelocities()

    def add_accept_or_reject_step(self):
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "x*accept + xold*(1-accept)")
        self.addComputePerDof("v", "v*accept - vold*(1-accept)")  # Notice the minus sign: momentum flip

    def add_accumulate_statistics_step(self):
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

    @property
    def kT(self):
        """The thermal energy."""
        return kB * self.temperature

    @property
    def b(self):
        """The scaling factor for preserving versus randomizing velocities."""
        return np.exp(-self.collision_rate * self.timestep)

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

    @property
    def effective_timestep(self):
        """The acceptance rate times the timestep."""
        return self.acceptance_rate * self.timestep


    def summary(self):
        """Return a dictionary of relevant state variables for XHMC, useful for debugging.
        Append self.summary() to a list and print out as a dataframe.
        """
        d = {}
        d["acceptance_rate"] = self.acceptance_rate
        d["effective_timestep"] = self.effective_timestep
        keys = ["accept", "ke", "Enew", "naccept", "ntrials", "Eold"]
        
        for key in keys:
            d[key] = self.getGlobalVariableByName(key)
        
        d["deltaE"] = d["Enew"] - d["Eold"]
        
        return d





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



class RandomTimestepGHMC(GHMC2):
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
        
        self.addGlobalVariable("scale_factor", 2.0) # Randomized scaling factor for timestep

        # Compute the thermal energy.
        self.temperature = temperature
        self.collision_rate = collision_rate        
        self.timestep = timestep
        self.create()

    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics."""
        print("Adding GHMC2 steps with randomization.")
        self.addComputeGlobal("scale_factor", "2.0 * uniform")        
        for step in range(self.steps_per_hmc):
            self.addComputePerDof("v", "v+0.5*dt*f/m * scale_factor")
            self.addComputePerDof("x", "x+dt*v * scale_factor")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dt*f/m * scale_factor+(x-x1)/(dt * scale_factor)")
            self.addConstrainVelocities()


class XHMCIntegrator(GHMC2):
    """Extra Chance Generalized Hamiltonian Monte Carlo."""
    def __init__(self, temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds, steps_per_hmc=10, k_max=2):
        """
        """
        mm.CustomIntegrator.__init__(self, timestep)
        
        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.collision_rate = collision_rate        
        self.timestep = timestep
        self.k_max = k_max
        
        self.create()


    def initialize_variables(self):

        self.addGlobalVariable("a", 1.0) # accept or reject
        self.addGlobalVariable("s", 0.0)
        self.addGlobalVariable("l", 0.0)
        self.addGlobalVariable("r", 0.0) # accept or reject

        self.addGlobalVariable("k_max", self.k_max)  # Maximum number of rounds of dynamics
        self.addGlobalVariable("k", 0)  # Current number of rounds of dynamics
        self.addGlobalVariable("kold", 0)  # Previous value of k stored for debugging purposes
        self.addGlobalVariable("flip", 0.0)  # Indicator variable whether this iteration was a flip
        
        self.addGlobalVariable("rho", 0.0)  # temporary variables for acceptance criterion
        self.addGlobalVariable("mu", 0.0)  # 
        self.addGlobalVariable("mu1", 0.0)  # 
        self.addGlobalVariable("mu10", 1.0) # Previous value of mu1
        self.addGlobalVariable("Uold", 0.0)
        self.addGlobalVariable("Unew", 0.0)
        self.addGlobalVariable("uni", 0.0)  # Uniform random variable generated once per round of XHMC

        self.addGlobalVariable("nflip", 0) # number of momentum flips (e.g. complete rejections)

        # Below this point is possible base class material
        
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addPerDofVariable("vold", 0) # old velocities
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        #self.addGlobalVariable("accept", 0) # DEFINED ABOVE
        self.addPerDofVariable("x1", 0) # for constraints
        self.addGlobalVariable("b", self.b) # velocity mixing parameter                

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""
        self.addComputeGlobal("s", "step(-k)")  # True only on first step of XHMC round
        self.addComputeGlobal("l", "step(k - k_max)")  # True only only last step of XHMC round

        self.addUpdateContextState()
        self.addConstrainPositions()

        self.addComputePerDof("v", "s * (sqrt(b) * v + (1 - s) * sqrt(1 - b) * sigma * gaussian) + (1 - s) * v")
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "s * (ke + energy) + (1 - s) * Eold")
        self.addComputeGlobal("Uold", "energy")  # Not strictly necessary
        self.addComputePerDof("xold", "s * x + (1 - s) * xold")
        self.addComputePerDof("vold", "s * v + (1 - s) * vold")
        
        self.addComputeGlobal("mu1", "mu1 * (1 - s)")
        self.addComputeGlobal("uni", "(1 - s) * uni + uniform * s")

    def add_accept_or_reject_step(self):
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")

        self.addComputeGlobal("Unew", "energy")
        self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
        self.addComputeGlobal("mu", "min(1, r)")
        self.addComputeGlobal("mu1", "max(mu1, mu)")
        self.addComputeGlobal("a", "step(mu1 - uni)")

        self.addComputeGlobal("flip", "(1 - a) * l")  # Flip is True ONLY on rejection at last cycle
        
        self.addComputePerDof("x", "x * a + xold * (1 - a)")
        self.addComputePerDof("v", "v * (1 - flip) - vold * flip")  # Conserve velocities except on flips.
        
        self.addComputeGlobal("kold", "k")  # Store the previous value of k for debugging purposes
        self.addComputeGlobal("k", "(k + 1) * (1 - flip) * (1 - a)")  # Increment by one ONLY if not flipping momenta or accepting, otherwise set to zero        

    def add_accumulate_statistics_step(self):
        self.addComputeGlobal("nflip", "nflip + flip")
        self.addComputeGlobal("naccept", "naccept + a")
        self.addComputeGlobal("ntrials", "ntrials + 1")

    @property
    def n_flip(self):
        """The total number of momentum flips."""
        return self.getGlobalVariableByName("nflip")

    @property
    def acceptance_rate(self):
        """The acceptance rate:"""
        return 1.0 - (self.k_max + 1) * self.n_flip / float(self.n_trials)

    def summary(self):
        """Return a dictionary of relevant state variables for XHMC, useful for debugging.
        Append self.summary() to a list and print out as a dataframe.
        """
        d = {}
        d["acceptance_rate"] = self.acceptance_rate
        keys = ["a", "s", "l", "rho", "ke", "Enew", "Unew", "mu", "mu1", "flip", "kold", "k", "naccept", "nflip", "ntrials", "Eold", "Uold"]
        for key in keys:
            d[key] = self.getGlobalVariableByName(key)
        
        d["deltaE"] = d["Enew"] - d["Eold"]
        
        return d






class GHMCRESPA(GHMC2):
    """Hybrid Monte Carlo (HMC) integrator with linearly ramped non-uniform timesteps.
    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, steps_per_hmc=10, timestep=1*simtk.unit.femtoseconds, collision_rate=91.0/simtk.unit.picoseconds, groups=None):
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

        if len(groups) == 0:
            raise ValueError("No force groups specified")
        
        self.groups = sorted(groups, key=lambda x: x[1])
        self.steps_per_hmc = steps_per_hmc

        self.collision_rate = collision_rate
        self.timestep = timestep
        self.temperature = temperature
        
        self.create()


    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics."""
        print("Adding GHMC RESPA steps.")
        for step in range(self.steps_per_hmc):
            self._create_substeps(1, self.groups)
            self.addConstrainVelocities()

    def _create_substeps(self, parentSubsteps, groups):
        
        group, substeps = groups[0]
        
        str_group, str_sub = str(group), str(substeps)
        
        stepsPerParentStep = substeps / parentSubsteps

        if stepsPerParentStep < 1 or stepsPerParentStep != int(stepsPerParentStep):
            raise ValueError("The number for substeps for each group must be a multiple of the number for the previous group")

        stepsPerParentStep = int(stepsPerParentStep) # needed for Python 3.x

        if group < 0 or group > 31:
            raise ValueError("Force group must be between 0 and 31")

        for i in range(stepsPerParentStep):
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))
            if len(groups) == 1:                
                self.addComputePerDof("x1", "x")
                self.addComputePerDof("x", "x+(dt/%s)*v" % (str_sub))
                self.addConstrainPositions()
                self.addComputePerDof("v", "(x-x1)/(dt/%s)" % (str_sub))
            else:
                self._create_substeps(substeps, groups[1:])
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))
