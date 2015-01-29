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
