"""
Infrastructure to define optimization problems.
"""
import math
import copy
import multiprocessing.dummy
from collections import OrderedDict


class Aborted(StopIteration):
    """Indicates abortion of optimization.

    May be due to an unpreventable problem without default treatment,
    e.g., a numerical problem. Inherits from :class:`StopIteration`.

    """
    def __init__(self, message=None):
        message_parts = ["Optimization aborted"]
        if message is not None:
            message_parts.extend([". Reason: ", message])
        StopIteration.__init__(self, "".join(message_parts))



class ResourcesExhausted(StopIteration):
    """Indicates an exhaustion of the specified budget (not an error).

    Inherits from :class:`StopIteration`.

    """
    def __init__(self, resource=None):
        message = "Resources exhausted"
        if resource is not None:
            message = "Resource exhausted: " + str(resource)
        StopIteration.__init__(self, message)



class Solved(StopIteration):
    """Indicates the problem has been satisfactorily solved.

    Inherits from :class:`StopIteration`.

    """
    def __init__(self, found_solution=None):
        self.found_solution = found_solution
        StopIteration.__init__(self, "Problem solved")



class Stalled(StopIteration):
    """Indicates stagnation of optimization (not an error).

    Inherits from :class:`StopIteration`.

    """
    def __init__(self, criterion=None):
        message_parts = ["Optimization has stalled"]
        if criterion is not None:
            message_parts.extend([". Criterion: ", str(criterion)])
        StopIteration.__init__(self, "".join(message_parts))



class Individual:
    """A data structure to store objective values together with the solution.

    Some methods of the problem classes expect objects with the two
    attributes `phenome` and `objective_values`. The exact type of these
    objects is irrelevant, but this class would be the obvious fit. The
    term 'phenome' stems from biology and means the whole set of phenotypic
    entities, in other words the form of appearance, of an animate being.
    So, it matches quite well as description of what is evaluated by the
    objective function.

    """
    def __init__(self, phenome=None, objective_values=None):
        """Constructor.

        Parameters
        ----------
        phenome : object, optional
            An arbitrary object which somehow can be evaluated by an
            objective function.
        objective_values : float or list, optional
            A single float or a list of objective values.

        """
        self.phenome = phenome
        self.objective_values = objective_values



class BundledObjectives:
    """Helper class to let several distinct functions appear as one."""

    def __init__(self, objective_functions):
        self.objective_functions = objective_functions


    def __call__(self, phenome):
        """Collect objective values from objective functions.

        Objective values are returned as flattened list.

        """
        returned_values = []
        for objective_function in self.objective_functions:
            returned_values.append(objective_function(phenome))
        flattened = []
        for returned_value in returned_values:
            try:
                iter(returned_value)
                # succeeded, so this function returned several values at once
                flattened.extend(returned_value)
            except TypeError:
                flattened.append(returned_value)
        return flattened



def identity(argument):
    return argument



class Problem(object):
    """The base class for problems to be solved.

    In the simplest case you can use this class directly by wrapping
    a single objective function or a list of objective functions. For
    more sophisticated cases, creating a subclass may be necessary.

    """
    def __init__(self, objective_functions,
                 num_objectives=None,
                 max_evaluations=float("inf"),
                 worker_pool=None,
                 mp_module=None,
                 phenome_preprocessor=None,
                 name=None):
        """Constructor.

        Parameters
        ----------
        objective_functions : callable or sequence of callables
            If this argument is simply a function, it is taken 'as-is'.
            If a sequence of callables is provided, these are wrapped in a
            :class:`BundledObjectives <optproblems.base.BundledObjectives>`
            helper object, so that a single function call returns a list of
            objective values.
        num_objectives : int, optional
            The number of objectives. If omitted, this number is guessed
            from the number of objective functions.
        max_evaluations : int, optional
            The maximum budget of function evaluations. By default there
            is no restriction.
        worker_pool : Pool, optional
            A pool of worker processes. Default is None (no 
            parallelization).
        mp_module : module, optional
            Either `multiprocessing`, `multiprocessing.dummy` (default),
            or a `MockMultiProcessing` instance. This is only used to create
            an internal lock around bookkeeping code in various places. The
            lock is only required for asynchronous parallelization, but not
            for the parallelization with a worker pool in 
            :func:`batch_evaluate`.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned. When
            this pre-processing raises an exception, no function
            evaluations are counted. By default, no pre-processing is
            applied.
        name : str, optional
            A nice name for humans to read.

        """
        try:
            iter(objective_functions)
            # succeeded, so several functions are given
            one_function = BundledObjectives(objective_functions)
            if num_objectives is None:
                # guess the number of objectives
                num_objectives = len(objective_functions)
            assert num_objectives >= len(objective_functions)
        except TypeError:
            one_function = objective_functions
            if num_objectives is None:
                num_objectives = 1
            assert num_objectives > 0
        self.objective_function = one_function
        self.num_objectives = num_objectives
        self.remaining_evaluations = max_evaluations
        self.consumed_evaluations = 0
        self.worker_pool = worker_pool
        self.chunksize = 1
        if mp_module is None:
            mp_module = multiprocessing.dummy
        self.mp_module = mp_module
        self.lock = mp_module.Manager().Lock()
        if phenome_preprocessor is None:
            phenome_preprocessor = identity
        self.phenome_preprocessor = phenome_preprocessor
        self.name = name


    def __str__(self):
        """Return the name of this problem."""
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__


    def __call__(self, phenome):
        """Evaluate a solution and return objective values.

        Also checks budget and counts the evaluation.

        Raises
        ------
        ResourcesExhausted
            If the budget of function evaluations is exhausted.

        """
        phenome = self.phenome_preprocessor(phenome)
        with self.lock:
            if self.remaining_evaluations > 0:
                self.consumed_evaluations += 1
                self.remaining_evaluations -= 1
            else:
                raise ResourcesExhausted("problem evaluations")
        objective_values = self.objective_function(phenome)
        try:
            num_obj_values = len(objective_values)
        except TypeError:
            num_obj_values = 1
        assert num_obj_values == self.num_objectives
        return objective_values


    def evaluate(self, individual):
        """Evaluate an individual.

        This method delegates the evaluation of the phenome to
        :func:`__call__ <optproblems.base.Problem.__call__>` and directly
        writes the objective values into the individual's corresponding
        attribute.

        """
        individual.objective_values = self.__call__(individual.phenome)


    def batch_evaluate(self, individuals):
        """Evaluate a batch of individuals.

        Objective values are written directly into the individuals'
        corresponding attributes.

        Raises
        ------
        ResourcesExhausted
            If the budget is not sufficient to evaluate all individuals,
            this exception is thrown.

        """
        if self.worker_pool is None or len(individuals) == 1:
            evaluate = self.evaluate
            for individual in individuals:
                evaluate(individual)
        else:
            preprocessor = self.phenome_preprocessor
            num_objectives = self.num_objectives
            with self.lock:
                budgeted_evaluations = min(len(individuals), self.remaining_evaluations)
                affordable_individuals = individuals[:budgeted_evaluations]
                phenomes = [preprocessor(ind.phenome) for ind in affordable_individuals]
                self.consumed_evaluations += budgeted_evaluations
                self.remaining_evaluations -= budgeted_evaluations
            results = self.worker_pool.map(self.objective_function,
                                           phenomes,
                                           chunksize=self.chunksize)
            for individual, objective_values in zip(individuals, results):
                try:
                    num_obj_values = len(objective_values)
                except TypeError:
                    num_obj_values = 1
                assert num_obj_values == num_objectives
                individual.objective_values = objective_values
            if len(individuals) > len(affordable_individuals):
                raise ResourcesExhausted("problem evaluations")



class MockMultiProcessing:
    """Mocks part of the interface of the multiprocessing module.

    This class provides no functionality except for calls to `Pool.map`,
    which are directed to the built-in :func:`map` function. Factory methods
    Manager(), Lock(), and Pool() all simply return this instance. The
    implemented interface is the subset required by the class
    :class:`Problem <optproblems.base.Problem>`. The motivation is to
    provide very lightweight code for problem instances that do not need
    parallelism or concurrency, and thus no synchronization.

    """
    @staticmethod
    def map(func, iterable, chunksize=1):
        """Emulate `Pool.map` by the built-in :func:`map` function."""
        return list(map(func, iterable))


    def close(self):
        pass


    def join(self):
        pass


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


    def Lock(self):
        return self


    def Manager(self):
        return self


    def Pool(self, processes=None, initializer=None, initargs=()):
        return self



class Cache(Problem):
    """Wrapper to save objective function evaluations of duplicates.

    This wrapper stores evaluations in an archive to reuse them for
    potentially occurring duplicates of previously seen solutions. This
    is especially useful in discrete search spaces.

    .. note:: This wrapper lets you limit the number of true
        evaluations, but not the number of 'virtual' ones. This may lead
        to infinite loops if your optimization algorithm contains bugs.
        For limiting the number of virtual evaluations, a construct such
        as ``Problem(Cache(Problem(...)), max_evaluations=x)`` would
        work.

    """
    def __init__(self, problem, hash_function=tuple):
        """Constructor.

        Parameters
        ----------
        problem : Problem
            A problem instance to be wrapped by this object.
        hash_function : callable, optional
            A function that gets the phenome as input and returns a
            hashable key for storing the objective values in a dictionary.
            By default we convert the phenome into a tuple to obtain this
            functionality.

        """
        self.problem = problem
        self.hash_function = hash_function
        self.saved_evaluations = 0
        self.archive = OrderedDict()


    @staticmethod
    def are_objectives_valid(objective_values):
        """Guess if a variable contains valid objective values.

        Objective values are only stored in the archive if this method
        returns True. In particular, None, strings, and empty lists are
        considered invalid. A list of arbitrary objects except for None
        and the empty list is considered valid, however. So, you may have
        to override this method if you have very exotic objective values.

        """
        valid = objective_values is not None
        try:
            num_obj_values = len(objective_values)
        except TypeError:
            return valid
        valid = valid and num_obj_values > 0
        try:
            valid = valid and None not in objective_values
            valid = valid and [] not in objective_values
        except TypeError:
            return False
        return valid


    @property
    def num_objectives(self):
        return self.problem.num_objectives


    @property
    def worker_pool(self):
        return self.problem.worker_pool


    @property
    def consumed_evaluations(self):
        return self.problem.consumed_evaluations


    @consumed_evaluations.setter
    def consumed_evaluations(self, value):
        self.problem.consumed_evaluations = value


    @property
    def remaining_evaluations(self):
        return self.problem.remaining_evaluations


    @remaining_evaluations.setter
    def remaining_evaluations(self, value):
        self.problem.remaining_evaluations = value


    def __str__(self):
        """Return the name of this problem."""
        return "Cached " + str(self.problem)


    def __call__(self, phenome):
        """Cached evaluation of one solution.

        Look up solution in archive, if not found delegate to original
        problem and store result in archive, if valid.

        """
        key = self.hash_function(phenome)
        recognized = False
        with self.problem.lock:
            if key in self.archive:
                objective_values = copy.deepcopy(self.archive[key])
                self.saved_evaluations += 1
                recognized = True
        if not recognized:
            objective_values = self.problem.__call__(phenome)
            if self.are_objectives_valid(objective_values):
                with self.problem.lock:
                    self.archive[key] = copy.deepcopy(objective_values)
        return objective_values


    def batch_evaluate(self, individuals):
        """Evaluate a batch of individuals, trying to avoid re-evaluations.

        If there is an entry for an individual in the archive, the values
        from there are reused as objective values. Else, the evaluation is
        delegated to the original problem. Finally, the new evaluations are
        stored in the archive, if they are valid.

        .. warning:: If ``worker_pool is not None``, duplicates in a batch
                    will not be recognized, leading to an expensive function
                    evaluation for each one.

        """
        if self.worker_pool is None or len(individuals) == 1:
            evaluate = self.evaluate
            for individual in individuals:
                evaluate(individual)
        else:
            hash_function = self.hash_function
            unrecognized_individuals = []
            # treat known solutions
            for individual in individuals:
                key = hash_function(individual.phenome)
                with self.problem.lock:
                    if key in self.archive:
                        individual.objective_values = copy.deepcopy(self.archive[key])
                        self.saved_evaluations += 1
                    else:
                        unrecognized_individuals.append(individual)
            # do actual evaluations
            try:
                self.problem.batch_evaluate(unrecognized_individuals)
            finally:
                are_objectives_valid = self.are_objectives_valid
                for individual in unrecognized_individuals:
                    valid = are_objectives_valid(individual.objective_values)
                    if valid:
                        key = hash_function(individual.phenome)
                        with self.problem.lock:
                            self.archive[key] = copy.deepcopy(individual.objective_values)



class ScalingPreprocessor:
    """Translation and linear transformation between arbitrary cuboids.

    This class is useful to normalize the search space, while the problem can
    keep using its "native" units.

    .. note:: This class makes use of the decorator design pattern for
        potential chaining of pre-processors, see
        https://en.wikipedia.org/wiki/Decorator_pattern

    """
    def __init__(self, from_cuboid, to_cuboid, previous_preprocessor=None):
        """Constructor.

        Parameters
        ----------
        from_cuboid : tuple of sequence
            This is the space an optimization algorithm would use. The first
            sequence contains the lower bounds, the second sequence
            contains the upper bounds for each variable.
        to_cuboid : tuple of sequence
            This is the domain of the optimization problem. The first
            sequence contains the lower bounds, the second sequence contains
            the upper bounds for each variable.
        previous_preprocessor : callable, optional
            Another callable that processes the phenome before this one
            does.

        """
        self.from_cuboid = from_cuboid
        self.to_cuboid = to_cuboid
        min_bounds_to, max_bounds_to = to_cuboid
        assert len(min_bounds_to) == len(max_bounds_to)
        for min_bound_to, max_bound_to in zip(min_bounds_to, max_bounds_to):
            assert max_bound_to >= min_bound_to
        min_bounds_from, max_bounds_from = from_cuboid
        assert len(min_bounds_from) == len(max_bounds_from)
        for min_bound_from, max_bound_from in zip(min_bounds_from, max_bounds_from):
            assert max_bound_from >= min_bound_from
        self.previous_preprocessor = previous_preprocessor


    def __call__(self, phenome):
        """Transform from one space to the other.

        This function does not check if the phenome is actually inside
        `from_cuboid`.

        """
        if self.previous_preprocessor is not None:
            phenome = self.previous_preprocessor(phenome)
        min_bounds_to, max_bounds_to = self.to_cuboid
        min_bounds_from, max_bounds_from = self.from_cuboid
        assert len(phenome) == len(min_bounds_from)
        # scale
        scaled_point = copy.copy(phenome)
        for i, value in enumerate(phenome):
            length_factor = max_bounds_to[i] - min_bounds_to[i]
            length_factor /= max_bounds_from[i] - min_bounds_from[i]
            scaled_point[i] = (phenome[i] - min_bounds_from[i]) * length_factor
            scaled_point[i] += min_bounds_to[i]
        return scaled_point



class BoundConstraintError(ValueError):
    """Used to report violations of bound constraints.

    Inherits from :class:`ValueError`.

    """
    def __init__(self, value, min_bound, max_bound, variable_name=None):
        bounds = "[" + str(min_bound) + ", " + str(max_bound) + "]"
        if variable_name is None:
            message = "Value " + str(value) + " out of bounds " + bounds
        else:
            message = "Variable " + variable_name + " (=" + str(value) + ") out of bounds " + bounds
        ValueError.__init__(self, message)



def min_bound_violated(value, min_bound):
    """Return True if ``min_bound != None and value < min_bound``."""
    return min_bound is not None and value < min_bound



def max_bound_violated(value, max_bound):
    """Return True if ``max_bound != None and value > max_bound``."""
    return max_bound is not None and value > max_bound



class BoundConstraintsChecker:
    """A pre-processor raising exceptions if bound constraints are violated.

    .. note:: This class makes use of the decorator design pattern for
        potential chaining of pre-processors, see
        https://en.wikipedia.org/wiki/Decorator_pattern

    """
    def __init__(self, bounds, previous_preprocessor=None):
        """Constructor.

        Parameters
        ----------
        bounds : tuple of sequence
            The first sequence contains the lower bounds, the second
            sequence contains the upper bounds for each variable.
        previous_preprocessor : callable, optional
            Another callable that processes the phenome before this one
            does.

        """
        self.min_bounds, self.max_bounds = bounds
        assert len(self.min_bounds) == len(self.max_bounds)
        self.previous_preprocessor = previous_preprocessor


    def __call__(self, phenome):
        """Check the bound constraints and raise exception if necessary.

        Raises
        ------
        BoundConstraintError
            If any bound constraint is violated.

        """
        if self.previous_preprocessor is not None:
            phenome = self.previous_preprocessor(phenome)
        max_bounds = self.max_bounds
        min_bounds = self.min_bounds
        assert len(phenome) == len(min_bounds)
        for i, phene in enumerate(phenome):
            if min_bound_violated(phene, min_bounds[i]) or max_bound_violated(phene, max_bounds[i]):
                raise BoundConstraintError(phene, min_bounds[i], max_bounds[i], variable_name="x"+str(i))
        return phenome



def project(value, min_bound, max_bound):
    """Clip the value to the feasible range.

    Parameters
    ----------
    value : float
        The decision variable to be repaired.
    min_bound : float
        Lower bound for `value`.
    max_bound : float
        Upper bound for `value`.

    Returns
    -------
    value : float
        The repaired decision variable.

    """
    if min_bound_violated(value, min_bound):
        return min_bound
    elif max_bound_violated(value, max_bound):
        return max_bound
    return value



def reflect(value, min_bound, max_bound):
    """Reflect the value between the two bounds until it lies inside them.

    Parameters
    ----------
    value : float
        The decision variable to be repaired.
    min_bound : float
        Lower bound for `value`.
    max_bound : float
        Upper bound for `value`.

    Returns
    -------
    value : float
        The repaired decision variable.

    """
    if math.isinf(value):
        raise Exception("Infinity detected in phenome")
    # carry out reflection
    if max_bound is not None and min_bound is not None:
        twice_the_range = 2 * (max_bound - min_bound)
        if abs(value - min_bound) > twice_the_range and max_bound != min_bound:
            # shortcut to save time, but next if-section still needed
            value = min_bound + math.fmod(abs(value - min_bound), twice_the_range)
    if max_bound is not None or min_bound is not None:
        # only take action if any boundary is not None
        if max_bound == min_bound:
            # the boundaries are equal and not None
            value = max_bound
        else:
            # while a boundary is violated, mirror at that boundary
            # this loop is finite because maxLim - minLim > 0
            is_max_violated = max_bound_violated(value, max_bound)
            is_min_violated = min_bound_violated(value, min_bound)
            while is_min_violated or is_max_violated:
                if is_max_violated:
                    value = max_bound - (value - max_bound)
                elif is_min_violated:
                    value = min_bound + (min_bound - value)
                is_max_violated = max_bound_violated(value, max_bound)
                is_min_violated = min_bound_violated(value, min_bound)
    return value



def wrap(value, min_bound, max_bound):
    """Treats the feasible space as a torus.

    Parameters
    ----------
    value : float
        The decision variable to be repaired.
    min_bound : float
        Lower bound for `value`.
    max_bound : float
        Upper bound for `value`.

    Returns
    -------
    value : float
        The repaired decision variable.

    """
    if max_bound is None or min_bound is None:
        raise Exception("Wrapping is only applicable if lower and upper bounds exist")
    if math.isinf(value):
        raise Exception("Infinity detected in phenome")
    # carry out wrapping
    if max_bound == min_bound:
        # the boundaries are equal and not None
        value = max_bound
    else:
        if min_bound_violated(value, min_bound) or max_bound_violated(value, max_bound):
            value = min_bound + math.fmod(value - min_bound, max_bound - min_bound)
        if min_bound_violated(value, min_bound):
            value += max_bound - min_bound
        elif max_bound_violated(value, max_bound):
            value += min_bound - max_bound
    return value



class BoundConstraintsRepair:
    """A pre-processor that repairs violations of bound constraints.

    More information about the available repair methods can be found
    in [Wessing2013]_.

    .. note:: This class makes use of the decorator design pattern for
        potential chaining of pre-processors, see
        https://en.wikipedia.org/wiki/Decorator_pattern

    References
    ----------
    .. [Wessing2013] S. Wessing (2013). Repair Methods for Box Constraints
        Revisited. In: Applications of Evolutionary Computation. Vol. 7835
        of Lecture Notes in Computer Science, pp. 469-478, Springer.
        https://dx.doi.org/10.1007/978-3-642-37192-9_47

    """
    def __init__(self, bounds, repair_modes, previous_preprocessor=None):
        """Constructor.

        Parameters
        ----------
        bounds : tuple of sequence
            The first sequence contains the lower bounds, the second
            sequence contains the upper bounds for each variable.
        repair_modes : sequence
            Contains an individual repair mode for each variable. The
            methods projection, reflection, and wrapping are supported.
            Values of None indicate that repair is not possible/desired.
            If a constraint violation is detected in this case, an
            exception is raised.
        previous_preprocessor : callable, optional
            Another callable that processes the phenome before this one
            does.

        """
        self.min_bounds, self.max_bounds = bounds
        assert len(self.min_bounds) == len(self.max_bounds)
        repair_modes = list(repair_modes)
        for i, repair_mode in enumerate(repair_modes):
            if repair_mode is None:
                repair_modes[i] = BoundConstraintError
            elif repair_mode in ("reflect", "reflection"):
                repair_modes[i] = reflect
            elif repair_mode in ("project", "projection"):
                repair_modes[i] = project
            elif repair_mode in ("wrap", "wrapping"):
                repair_modes[i] = wrap
            elif not callable(repair_mode):
                raise Exception("Unknown repair mode: " + str(repair_mode))
        self.repair_modes = repair_modes
        assert len(self.repair_modes) == len(self.min_bounds)
        self.previous_preprocessor = previous_preprocessor


    def __call__(self, phenome):
        """Return a repaired version of this phenome.

        Does not modify the original.

        """
        max_bounds = self.max_bounds
        min_bounds = self.min_bounds
        if self.previous_preprocessor is not None:
            phenome = self.previous_preprocessor(phenome)
        phenome = copy.deepcopy(phenome)
        repair_modes = self.repair_modes
        # determine repair mode and execute
        for i, phene in enumerate(phenome):
            repair_mode = repair_modes[i]
            phenome[i] = repair_mode(phene, min_bounds[i], max_bounds[i])
        return phenome



class TestProblem(Problem):
    """Abstract base class for artificial test problems."""

    def get_optimal_solutions(self, max_number=None):
        """Return globally optimal or Pareto-optimal solutions.

        This is an abstract method. Implementations must be deterministic.
        In the multi-objective case, the generated solutions should be
        evenly distributed over the whole Pareto-set.

        """
        raise NotImplementedError("Optimal solutions are unknown.")


    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions.

        This is an abstract method. Implementations must be deterministic.
        This method should be most useful for single-objective, continuous
        problems.

        """
        raise NotImplementedError("Locally optimal solutions are unknown.")
