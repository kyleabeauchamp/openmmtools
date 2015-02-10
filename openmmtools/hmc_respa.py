import simtk.unit
import simtk.openmm as mm
import numpy as np

kB = simtk.unit.BOLTZMANN_CONSTANT_kB * simtk.unit.AVOGADRO_CONSTANT_NA


def guess_force_groups(system):
    for force in system.getForces():
        if isinstance(force, mm.openmm.NonbondedForce):
            force.setForceGroup(1)
            force.setReciprocalSpaceForceGroup(2)


class RampedHMCRespaIntegrator(simtk.openmm.CustomIntegrator):
    """
    Hybrid Monte Carlo (HMC) integrator.

    """

    def __init__(self, groups, temperature=298.0*simtk.unit.kelvin, nsteps=10, timestep=1*simtk.unit.femtoseconds, max_boost=0.1):
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

        super(RampedHMCRespaIntegrator, self).__init__(timestep)

        if len(groups) == 0:
            raise ValueError("No force groups specified")
        
        groups = sorted(groups, key=lambda x: x[1])


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
        
        for step in range(nsteps):
            rho = rho_grid[step]
            self.addUpdateContextState()
            self._create_substeps(1, groups, rho)
            self.addConstrainVelocities()

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

            
    def _create_substeps(self, parentSubsteps, groups, rho):
        
        group, substeps = groups[0]
        
        str_group, str_sub = str(group), str(substeps)
        
        stepsPerParentStep = substeps / parentSubsteps

        if stepsPerParentStep < 1 or stepsPerParentStep != int(stepsPerParentStep):
            raise ValueError("The number for substeps for each group must be a multiple of the number for the previous group")

        stepsPerParentStep = int(stepsPerParentStep) # needed for Python 3.x

        if group < 0 or group > 31:
            raise ValueError("Force group must be between 0 and 31")

        for i in range(stepsPerParentStep):
            self.addComputePerDof("v", "v+%f*0.5*(dt/%s)*f%s/m" % (rho, str_sub, str_group))
            if len(groups) == 1:
                self.addComputePerDof("x1", "x")
                self.addComputePerDof("x", "x+%f*(dt/%s)*v" % (rho, str_sub))
                self.addConstrainPositions()
                self.addComputePerDof("v", "(x-x1)/(%f*dt/%s)" % (rho, str_sub))
            else:
                self._create_substeps(substeps, groups[1:], rho)
            self.addComputePerDof("v", "v+%f*0.5*(dt/%s)*f%s/m" % (rho, str_sub, str_group))


class HMCRespaIntegrator(simtk.openmm.CustomIntegrator):
    """
    Hybrid Monte Carlo (HMC) integrator.

    """

    def __init__(self, groups, temperature=298.0*simtk.unit.kelvin, nsteps=10, timestep=1*simtk.unit.femtoseconds):
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

        super(HMCRespaIntegrator, self).__init__(timestep)

        if len(groups) == 0:
            raise ValueError("No force groups specified")
        
        groups = sorted(groups, key=lambda x: x[1])


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

        for step in range(nsteps):
            self.addUpdateContextState()
            self._create_substeps(1, groups)
            self.addConstrainVelocities()

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


class GHMCIntegrator2(simtk.openmm.CustomIntegrator):
    """
    Generalized hybrid Monte Carlo (GHMC) integrator.

    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds, nsteps=10):
        """
        Create a generalized hybrid Monte Carlo (GHMC) integrator.

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

        # Initialize constants.
        kT = kB * temperature
        gamma = collision_rate

        # Create a new custom integrator.
        super(GHMCIntegrator2, self).__init__(timestep)
        print(gamma, timestep, -gamma*timestep)
        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT) # thermal energy
        self.addGlobalVariable("b", np.exp(-gamma*timestep)) # velocity mixing parameter
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("vold", 0) # old velocities
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials
        self.addPerDofVariable("x1", 0) # position before application of constraints

        #
        # Pre-computation.
        # This only needs to be done once, but it needs to be done for each degree of freedom.
        # Could move this to initialization?
        #
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Allow context updating here.
        #
        self.addUpdateContextState();

        #
        # Constrain positions.
        #
        self.addConstrainPositions();

        #
        # Velocity perturbation.
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities();

        #
        # Metropolized symplectic step.
        #
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        
        for step in range(nsteps):
            self.addComputePerDof("v", "v + 0.5*dt*f/m")
            self.addComputePerDof("x", "x + v*dt")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions();
            self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
            self.addConstrainVelocities();

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "x*accept + xold*(1-accept)")
        self.addComputePerDof("v", "v*accept - vold*(1-accept)")

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
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
    def acceptance_rate(self):
        """The acceptance rate: n_accept  / n_trials."""
        return self.n_accept / float(self.n_trials)


class XHMCIntegrator(simtk.openmm.CustomIntegrator):
    """
    Generalized X hybrid Monte Carlo (XGHMC) integrator.

    """

    def add_hmc_loop(self, nsteps):
        """Propagate nsteps iterations of HMC."""
        for step in range(nsteps):
            self.addComputePerDof("v", "v + 0.5*dt*f/m")
            self.addComputePerDof("x", "x + v*dt")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
            self.addConstrainVelocities()

    def __init__(self, temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds, nsteps=10, k_max=2):
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

        # Initialize constants.
        kT = kB * temperature
        gamma = collision_rate
        
        self.k_max = k_max

        # Create a new custom integrator.
        super(XHMCIntegrator, self).__init__(timestep)
        print(gamma, timestep, -gamma*timestep)
        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT) # thermal energy
        self.addGlobalVariable("b", np.exp(-gamma*timestep)) # velocity mixing parameter
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("vold", 0) # old velocities
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("a", 1.0) # accept or reject
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials
        self.addGlobalVariable("nflip", 0) # number of momentum flips (e.g. complete rejections)
        self.addPerDofVariable("x1", 0) # position before application of constraints


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

        self.addComputePerDof("sigma", "sqrt(kT/m)")        
        
        self.add_computations(nsteps)
        
    def add_computations(self, nsteps):

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

        self.add_hmc_loop(nsteps)

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
        """The acceptance rate: 1 - n_flip / n_trials"""
        return 1.0 - self.k_max * self.n_flip / float(self.n_trials)
