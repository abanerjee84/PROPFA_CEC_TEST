"""
This module contains miscellaneous test problems with continuous/real-valued
search space. The problems are mostly from the early days of research on
optimization.

"""
import math
import random
import itertools

from base import TestProblem, BoundConstraintsChecker
from base import Individual


TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0


class SequenceChecker:
    """A pre-processor raising exceptions if the phenome has the wrong length.

    .. note:: This class makes use of the decorator design pattern for
        potential chaining of pre-processors, see
        https://en.wikipedia.org/wiki/Decorator_pattern

    """
    def __init__(self, num_variables, data_type=None, previous_preprocessor=None):
        """Constructor.

        Parameters
        ----------
        num_variables : int
            The expected number of variables.
        data_type : class, optional
            If given, it is tested if all elements belong to this type.
        previous_preprocessor : callable, optional
            Another callable that processes the phenome before this one
            does.

        """
        self.num_variables = num_variables
        self.data_type = data_type
        self.previous_preprocessor = previous_preprocessor


    def __call__(self, phenome):
        """Check the phenome and raise exception if necessary.

        Raises
        ------
        Exception
            If the length or data type is wrong.

        """
        if self.previous_preprocessor is not None:
            phenome = self.previous_preprocessor(phenome)
        assert len(phenome) == self.num_variables
        data_type = self.data_type
        if data_type is not None:
            for phene in phenome:
                assert isinstance(phene, data_type)
        return phenome



class Shekel(TestProblem):
    """Shekel's family of test problems.

    As defined in [Dixon1978]_. The problems have four variables with lower
    and upper bounds of 0 and 10, respectively.

    """

    def __init__(self, num_optima, phenome_preprocessor=None, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_optima : int
            The number of local optima. Must be between 1 and 10.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        assert num_optima > 0 and num_optima <= 10
        self.num_optima = num_optima
        self.num_variables = 4
        self._min_bounds = [0.0] * self.num_variables
        self._max_bounds = [10.0] * self.num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self,
                             self.objective_function,
                             num_objectives=1,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.a = [[4.0, 4.0, 4.0, 4.0],
                  [1.0, 1.0, 1.0, 1.0],
                  [8.0, 8.0, 8.0, 8.0],
                  [6.0, 6.0, 6.0, 6.0],
                  [3.0, 7.0, 3.0, 7.0],
                  [2.0, 9.0, 2.0, 9.0],
                  [5.0, 5.0, 3.0, 3.0],
                  [8.0, 1.0, 8.0, 1.0],
                  [6.0, 2.0, 6.0, 2.0],
                  [7.0, 3.6, 7.0, 3.6]][:num_optima]
        self.c = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5][:num_optima]
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def objective_function(self, phenome):
        num_variables = self.num_variables
        assert num_variables == len(phenome)
        a = self.a
        ret = 0.0
        for i in range(self.num_optima):
            diff_vector = [phenome[j] - a[i][j] for j in range(num_variables)]
            sum_of_squares = sum(diff ** 2 for diff in diff_vector)
            ret -= 1.0 / (sum_of_squares + self.c[i])
        return ret


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        local_optima = self.get_locally_optimal_solutions()
        objective_values = [self.objective_function(opt.phenome) for opt in local_optima]
        minimum = float("inf")
        opt_solutions = []
        for obj_value, individual in zip(objective_values, local_optima):
            if obj_value == minimum:
                opt_solutions.append(individual)
            elif obj_value < minimum:
                opt_solutions = [individual]
                minimum = obj_value
        return opt_solutions


    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        approx_optima = self.a
        rand_gen = random.Random()
        rand_gen.seed(2)
        optima = []
        for approx_opt in approx_optima:
            current_opt = approx_opt
            current_opt_objective = self.objective_function(current_opt)
            for _ in range(10000):
                lower = [x - 0.0001 for x in current_opt]
                upper = [x + 0.0001 for x in current_opt]
                rand_point = []
                for low, high in zip(lower, upper):
                    next_value = rand_gen.uniform(low, high)
                    next_value += rand_gen.uniform(low, high)
                    rand_point.append(next_value * 0.5)
                is_feasible = True
                for i in range(len(rand_point)):
                    is_feasible &= rand_point[i] >= self.min_bounds[i]
                    is_feasible &= rand_point[i] <= self.max_bounds[i]
                if is_feasible:
                    obj_value = self.objective_function(rand_point)
                    if obj_value < current_opt_objective:
                        current_opt_objective = obj_value
                        current_opt = rand_point
            optima.append(Individual(list(current_opt)))
        if max_number is not None:
            optima = optima[:max_number]
        return optima



class Hartman3(TestProblem):
    """A 3-D instance of Hartman's family.

    The principle for defining problems of this family was presented in
    [Hartman1972]_. The numbers for this instance can be found in
    [Dixon1978]_. The search space is the unit hypercube.

    References
    ----------
    .. [Hartman1972] Hartman, James K. (1972). Some Experiments in Global
        Optimization. Technical report NP5 55HH72051A, Naval Postgraduate
        School, Monterey, California.

    """
    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 3
        self._min_bounds = [0.0] * self.num_variables
        self._max_bounds = [1.0] * self.num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, self.objective_function,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.a = [[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]]
        self.p = [[0.3689, 0.1170, 0.2673],
                  [0.4699, 0.4387, 0.7470],
                  [0.1091, 0.8732, 0.5547],
                  [0.03815, 0.5743, 0.8828]]
        self.c = [1.0, 1.2, 3.0, 3.2]


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def objective_function(self, phenome):
        num_variables = self.num_variables
        assert len(phenome) == num_variables
        ret = 0.0
        p = self.p
        a = self.a
        for i in range(4):
            temp_sum = sum(a[i][j] * (phenome[j] - p[i][j]) ** 2 for j in range(num_variables))
            ret -= self.c[i] * math.exp(-temp_sum)
        return ret


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        local_optima = self.get_locally_optimal_solutions()
        objective_values = [self.objective_function(opt.phenome) for opt in local_optima]
        minimum = float("inf")
        opt_solutions = []
        for obj_value, individual in zip(objective_values, local_optima):
            if obj_value == minimum:
                opt_solutions.append(individual)
            elif obj_value < minimum:
                opt_solutions = [individual]
                minimum = obj_value
        return opt_solutions


    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions.

        According to [Toern1999]_, this problem has four local optima.
        However, only three could be found experimentally.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        References
        ----------
        .. [Toern1999] A. Toern; M.M. Ali; S. Viitanen (1999). Stochastic
            Global Optimization: Problem Classes and Solution Techniques.
            Journal of Global Optimization, vol. 14, pp. 437-447.

        """
        optima = []
        phenomes = [[0.36872272, 0.11756162, 0.26757374],
                    [0.10933749, 0.86052422, 0.56412316],
                    [0.11461436, 0.55564884, 0.85254695]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max_number]
        return optima



class Hartman6(TestProblem):
    """A 6-D instance of Hartman's family.

    The principle for defining problems of this family was presented in
    [Hartman1972]_. The numbers for this instance can be found in
    [Dixon1978]_. The search space is the unit hypercube.

    """
    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 6
        self._min_bounds = [0.0] * self.num_variables
        self._max_bounds = [1.0] * self.num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, self.objective_function,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.a = [[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]]
        self.p = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
        self.c = [1.0, 1.2, 3.0, 3.2]


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def objective_function(self, phenome):
        num_variables = self.num_variables
        assert num_variables == len(phenome)
        ret = 0.0
        p = self.p
        a = self.a
        for i in range(4):
            temp_sum = sum(a[i][j] * (phenome[j] - p[i][j]) ** 2 for j in range(num_variables))
            ret -= self.c[i] * math.exp(-temp_sum)
        return ret


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        local_optima = self.get_locally_optimal_solutions()
        objective_values = [self.objective_function(opt.phenome) for opt in local_optima]
        minimum = float("inf")
        opt_solutions = []
        for obj_value, individual in zip(objective_values, local_optima):
            if obj_value == minimum:
                opt_solutions.append(individual)
            elif obj_value < minimum:
                opt_solutions = [individual]
                minimum = obj_value
        return opt_solutions


    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions.

        According to [Toern1999]_, this problem has four local optima.
        However, only two could be found experimentally.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        optima = []
        phenomes = [[0.20168951, 0.15001068, 0.47687397, 0.27533242, 0.31165161, 0.65730053],
                    [0.40465312, 0.88244492, 0.84610156, 0.57398968, 0.13892656, 0.03849589]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max_number]
        return optima



def branin(phenome):
    """The bare-bones Branin function."""
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    f = 1.0 / (8.0 * math.pi)
    ret = (phenome[1] - b * phenome[0] ** 2 + c * phenome[0] - 6.0) ** 2
    ret += 10.0 * (1.0 - f) * math.cos(phenome[0]) + 10.0
    return ret



class Branin(TestProblem):
    """Branin's test problem 'RCOS'.

    The search space is :math:`[-5, 0] \\times [10, 15]`. Every optimum is
    a global optimum.

    """

    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 2
        self._min_bounds = [-5.0, 0.0]
        self._max_bounds = [10.0, 15.0]
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, branin,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the three global optima."""
        optima = []
        phenomes = [[-math.pi, 12.275], [math.pi, 2.275], [9.424778, 2.475]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max_number]
        return optima


    get_locally_optimal_solutions = get_optimal_solutions



def goldstein_price(phenome):
    """The bare-bones Goldstein-Price function."""
    x = phenome[0]
    y = phenome[1]
    long_part1 = 19.0 - 14.0 * x + 3.0 * x ** 2 - 14.0 * y
    long_part1 += 6.0 * x * y + 3.0 * y ** 2
    long_part1 *= (x + y + 1.0) ** 2
    long_part2 = 18.0 - 32.0 * x + 12.0 * x ** 2
    long_part2 += 48.0 * y - 36.0 * x * y + 27.0 * y ** 2
    long_part2 *= (2.0 * x - 3.0 * y) ** 2
    return (1.0 + long_part1) * (30.0 + long_part2)



class GoldsteinPrice(TestProblem):
    """A test problem by Goldstein and Price.

    The search space is :math:`[-2, 2] \\times [-2, 2]`.

    """

    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 2
        self._min_bounds = [-2.0] * self.num_variables
        self._max_bounds = [2.0] * self.num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, goldstein_price,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0, -1.0])]


    def get_locally_optimal_solutions(self, max_number=None):
        """Return the four locally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        optima = []
        phenomes = [[0.0, -1.0], [-0.6, -0.4], [1.2, 0.8], [1.8, 0.2]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max_number]
        return optima



class DixonSzegoe(list):
    """The test problem collection of Dixon and Szegoe for global optimization.

    This class inherits from :class:`list` and fills itself with the seven
    problems Shekel5, Shekel7, Shekel10, Hartman3, Hartman6, Branin, and
    GoldsteinPrice. The arguments to the constructor are passed through to
    the problem classes.

    References
    ----------
    .. [Dixon1978] L.C.W. Dixon and G.P. Szegoe, The global optimization
        problem: an introduction, pp. 1-15 in: in L.C.W. Dixon and G.P.
        Szegoe (eds.), Towards Global Optimisation 2, North-Holland,
        Amsterdam 1978.

    """
    def __init__(self, **kwargs):
        shekel5 = Shekel(5, name="Shekel5", **kwargs)
        shekel7 = Shekel(7, name="Shekel7", **kwargs)
        shekel10 = Shekel(10, name="Shekel10", **kwargs)
        hart3 = Hartman3(**kwargs)
        hart6 = Hartman6(**kwargs)
        brn = Branin(**kwargs)
        gp = GoldsteinPrice(**kwargs)
        list.__init__(self, [shekel5, shekel7, shekel10, hart3, hart6, brn, gp])



def ackley(phenome):
    """The bare-bones Ackley function."""
    num_variables = len(phenome)
    a = 20.0
    b = 0.2
    sum1 = 0.0
    sum2 = 0.0
    for i in range(num_variables):
        sum1 += phenome[i] ** 2
        sum2 += math.cos(TWO_PI * phenome[i])
    value = -a * math.exp(-b * math.sqrt(1.0 / num_variables * sum1))
    value += -math.exp(1.0 / num_variables * sum2) + a + math.e
    return value



class Ackley(TestProblem):
    """Ackley's test problem.

    No bound constraints are pre-defined for this problem.

    """
    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, ackley,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



def double_sum(phenome):
    """Schwefel's problem 1.2"""
    ret = 0.0
    slice_sum = 0.0
    for x in phenome:
        slice_sum += x
        ret += slice_sum ** 2
    return ret



class DoubleSum(TestProblem):
    """Schwefel's double-sum problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, double_sum,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



class EllipsoidFunction:
    """A configurable Ellipsoid function.

    The basic one-dimensional formula reads ``a ** exponent * x ** 2``.

    """
    def __init__(self, a=1.0e6):
        self.a = a


    def __call__(self, phenome):
        """Evaluate the function."""
        num_variables = len(phenome)
        result = 0.0
        for i in range(num_variables):
            exponent = float(i) / (num_variables - 1)
            result += self.a ** exponent * phenome[i] ** 2
        return result



class Ellipsoid(TestProblem):
    """A configurable ellipsoidal test problem.

    No bound constraints are pre-defined for this problem.

    """
    def __init__(self, num_variables=30, a=1.0e6, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, EllipsoidFunction(a),
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



class FletcherPowell(TestProblem):
    """Fletcher and Powell's test problem.

    Each decision variable is restricted to :math:`[-\\pi, \\pi]` and the
    search space is periodic.

    References
    ----------
    .. [Fletcher1963] R. Fletcher and M. J. D. Powell (1963). A Rapidly
        Convergent Descent Method for Minimization. The Computer Journal
        6(2): 163-168, https://dx.doi.org/10.1093/comjnl/6.2.163

    """
    def __init__(self, num_variables=10,
                 rand_gen=None,
                 phenome_preprocessor=None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            The number of decision variables.
        rand_gen : random.Random, optional
            A generator for random numbers. If omitted, the global instance
            of the module :mod:`random` is used.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        if rand_gen is None:
            rand_gen = random
        self.num_variables = num_variables
        self._min_bounds = [-math.pi] * num_variables
        self._max_bounds = [math.pi] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, self.objective_function,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.alpha = [rand_gen.uniform(-math.pi, math.pi) for _ in range(num_variables)]
        self.a = [[rand_gen.randint(-100, 100) for _ in range(num_variables)] for _ in range(num_variables)]
        self.b = [[rand_gen.randint(-100, 100) for _ in range(num_variables)] for _ in range(num_variables)]
        self.init_vector_e()


    def init_vector_e(self):
        self.e = [0.0] * self.num_variables
        for i in range(self.num_variables):
            self.e[i] = 0.0
            for j in range(self.num_variables):
                self.e[i] += self.a[i][j] * math.sin(self.alpha[j])
                self.e[i] += self.b[i][j] * math.cos(self.alpha[j])


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual(list(self.alpha))]


    def objective_function(self, phenome):
        # shortcuts & initialization
        a = self.a
        b = self.b
        e = self.e
        sin = math.sin
        cos = math.cos
        ret = 0.0
        lhs = [0.0] * self.num_variables
        # calculate
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                lhs[i] += a[i][j] * sin(phenome[j]) + b[i][j] * cos(phenome[j])
            ret += (e[i] - lhs[i]) ** 2
        return ret



def griewank(phenome):
    """The bare-bones Griewank function."""
    ssum = 0.0
    product = 1.0
    for i in range(len(phenome)):
        ssum += phenome[i] ** 2 / 4000.0
        product *= math.cos(phenome[i] / math.sqrt(i + 1.0))
    return ssum - product + 1.0



class Griewank(TestProblem):
    """Griewank's test problem.

    No bound constraints are pre-defined for this problem. A possible choice
    is :math:`[-600, 600]` for each variable.

    """
    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, griewank,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



def himmelblau(phenome):
    """The bare-bones Himmelblau function."""
    x = phenome[0]
    y = phenome[1]
    return (x ** 2 + y - 11.0) ** 2 + (x + y ** 2 - 7.0) ** 2



class Himmelblau(TestProblem):
    """Himmelblau's test problem.

    No bound constraints are pre-defined for this problem. Possible choices
    including all the optima are :math:`[-4, 4] \\times [-4, 4]` or
    larger rectangles.

    References
    ----------
    .. [Himmelblau1972] David M. Himmelblau, Applied Nonlinear Programming,
        McGraw Hill, 1972

    """
    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 2
        preprocessor = SequenceChecker(self.num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, himmelblau,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the four optimal solutions.

        All local optima are global optima in this problem.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        optima = []
        phenomes = [[3.0, 2.0],
                    [-3.779310253377747, -3.283185991286170],
                    [-2.805118086952745, 3.131312518250573],
                    [3.584428340330492, -1.848126526964404]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max(1, max_number)]
        return optima


    get_locally_optimal_solutions = get_optimal_solutions



class LunacekTwoSpheres(TestProblem):
    """Lunacek's two spheres.

    References
    ----------
    .. [Lunacek2008] M. Lunacek, D. Whitley, and A. Sutton (2008). The
        Impact of Global Structure on Search. In: Parallel Problem Solving
        from Nature, Lecture Notes in Computer Science, vol. 5199,
        pp. 498-507, Springer.

    """
    def __init__(self, num_variables=10,
                 depth=0.0,
                 size=1.0,
                 phenome_preprocessor=None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            Number of decision variables of the problem.
        depth : float, optional
            Depth parameter of the worse basin.
        size : float, optional
            Size parameter of the worse basin.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        self.depth = depth
        self.size = size
        self.offset1 = 2.5
        self.offset2 = -math.sqrt((self.offset1 ** 2 - depth) / size)
        self.num_variables = num_variables
        self._min_bounds = [-5.0] * num_variables
        self._max_bounds = [5.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, self.objective_function,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def objective_function(self, phenome):
        # shortcuts
        shifted1 = [phene - self.offset1 for phene in phenome]
        shifted2 = [phene - self.offset2 for phene in phenome]
        value1 = sphere(shifted1)
        value2 = self.depth * self.num_variables + self.size * sphere(shifted2)
        return min(value1, value2)


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        optima = []
        phenomes = [[self.offset1] * self.num_variables]
        if self.depth == 0.0:
            phenomes.append([self.offset2] * self.num_variables)
        elif self.depth < 0.0:
            phenomes = [[self.offset2] * self.num_variables]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max(1, max_number)]
        return optima


    def get_locally_optimal_solutions(self, max_number=None):
        """Return the locally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        optima = []
        phenomes = [[self.offset1] * self.num_variables,
                    [self.offset2] * self.num_variables]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max(1, max_number)]
        return optima



class LunacekTwoRastrigins(LunacekTwoSpheres):
    """Lunacek's two Rastrigin functions."""

    def __init__(self, num_variables=10,
                 depth=0.0,
                 size=1.0,
                 a=10.0,
                 omega=TWO_PI,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            Number of decision variables of the problem.
        depth : float, optional
            Depth parameter of the worse basin.
        size : float, optional
            Size parameter of the worse basin.
        a : float, optional
            Amplitude of the cosine term of the rastrigin function.
        omega : float, optional
            Controls the period length of the cosine term of the rastrigin
            function.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        LunacekTwoSpheres.__init__(self, num_variables,
                                   depth,
                                   size,
                                   **kwargs)
        self.a = a
        self.omega = omega


    def objective_function(self, phenome):
        # shortcuts
        a = self.a
        omega = self.omega
        # calculate
        sphere_obj_value = LunacekTwoSpheres.objective_function(self, phenome)
        shifted1 = [phene - self.offset1 for phene in phenome]
        rastrigin_part = a * len(shifted1)
        for x in shifted1:
            rastrigin_part -= a * math.cos(omega * x)
        return sphere_obj_value + rastrigin_part


    def get_locally_optimal_solutions(self, max_number=None):
        raise NotImplementedError("Locally optimal solutions are unknown.")



class ModifiedRastriginFunction:
    """A function similar to the Rastrigin function.

    The basic one-dimensional formula reads
    ``2.0 * k * x ** 2 + 10.0 * cos(omega * x)``. Further information can
    be found in [Saha2010]_.

    """
    def __init__(self, num_variables, omegas, k_values):
        self.num_variables = num_variables
        if omegas is None:
            omegas = [TWO_PI] * num_variables
        self.omegas = omegas
        self.k_values = k_values


    def __call__(self, phenome):
        """Evaluate the function."""
        # shortcuts
        omegas = self.omegas
        k_values = self.k_values
        cos = math.cos
        # calculate
        ret = 10.0 * self.num_variables
        for i, x in enumerate(phenome):
            ret += 10.0 * cos(omegas[i] * x) + 2.0 * k_values[i] * x ** 2
        return ret



class ModifiedRastrigin(TestProblem):
    """A test problem similar to the Rastrigin problem.

    The modification consists of a configurable number of local optima per
    dimension, so that the total number of local optima becomes less
    dependent on the dimension. The problem was defined in [Saha2010]_.
    There are three pre-defined instances with 4, 8, and 16 variables,
    respectively, which can be obtained from the
    :func:`create_instance <optproblems.real.ModifiedRastrigin.create_instance>`
    method. The search space is the unit hypercube.

    References
    ----------
    .. [Saha2010] Amit Saha, Kalyanmoy Deb (2010). A Bi-criterion Approach
        to Multimodal Optimization: Self-adaptive Approach. In: Simulated
        Evolution and Learning, vol. 6457 of Lecture Notes in Computer
        Science, pp. 95-104, Springer

    """
    opt_x_for_k = [[],
                   [0.494984],
                   [0.24874, 0.74622],
                   [0.16611, 0.49832, 0.83053],
                   [0.12468, 0.37405, 0.62342, 0.87279]]

    def __init__(self, num_variables=16,
                 k_values=None,
                 phenome_preprocessor=None,
                 **kwargs):
        if k_values is None:
            k_values = [1] * num_variables
        self.k_values = k_values
        omegas = [TWO_PI * k for k in k_values]
        self._min_bounds = [0.0] * num_variables
        self._max_bounds = [1.0] * num_variables
        self.num_variables = num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        obj_function = ModifiedRastriginFunction(num_variables, omegas, k_values)
        TestProblem.__init__(self, obj_function,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    @staticmethod
    def create_instance(num_variables, **kwargs):
        """Factory method for pre-defined modified Rastrigin problems.

        Parameters
        ----------
        num_variables : int
            Must be 4, 8, or 16.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor.

        Returns
        -------
        problem : ModifiedRastrigin instance

        """
        if num_variables == 4:
            k_values = [2, 2, 3, 4]
        elif num_variables == 8:
            k_values = [1, 2, 1, 2, 1, 3, 1, 4]
        elif num_variables == 16:
            k_values = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]
        else:
            raise Exception("There is no predefined instance for " + str(num_variables) + " variables")
        return ModifiedRastrigin(num_variables, k_values, **kwargs)


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        local_optima = self.get_locally_optimal_solutions()
        objective_values = [self.objective_function(opt.phenome) for opt in local_optima]
        minimum = float("inf")
        opt_solutions = []
        for obj_value, individual in zip(objective_values, local_optima):
            if obj_value == minimum:
                opt_solutions.append(individual)
            elif obj_value < minimum:
                opt_solutions = [individual]
                minimum = obj_value
        if max_number is not None:
            opt_solutions = opt_solutions[:max(1, max_number)]
        return opt_solutions


    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        positions = []
        for i in range(self.num_variables):
            positions.append(self.opt_x_for_k[self.k_values[i]])
        optima = []
        if max_number is None:
            max_number = float("inf")
        # build cross product of all dimensions
        for position in itertools.product(*positions):
            optima.append(Individual(list(position)))
            if len(optima) >= max_number:
                break
        return optima



class RastriginFunction:
    """A configurable Rastrigin function.

    The basic one-dimensional formula reads ``x ** 2 - a * cos(omega * x)``.

    """
    def __init__(self, a=10.0, omega=TWO_PI):
        self.a = a
        self.omega = omega


    def __call__(self, phenome):
        """Evaluate the function."""
        a = self.a
        omega = self.omega
        cos = math.cos
        ret = a * len(phenome)
        for x in phenome:
            ret += x ** 2 - a * cos(omega * x)
        return ret



class Rastrigin(TestProblem):
    """A configurable Rastrigin test problem.

    No bound constraints are pre-defined for this problem, but
    :math:`[-5, 5]` for every variable is a typical choice.

    """
    def __init__(self, num_variables=10,
                 a=10.0,
                 omega=TWO_PI,
                 phenome_preprocessor=None,
                 **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, RastriginFunction(a, omega),
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



def rosenbrock(phenome):
    """The bare-bones Rosenbrock function."""
    ret = 0.0
    for i in range(len(phenome) - 1):
        x = phenome[i]
        ret += 100.0 * (x ** 2 - phenome[i+1]) ** 2 + (x - 1.0) ** 2
    return ret



class Rosenbrock(TestProblem):
    """Rosenbrock's test problem.

    No bound constraints are pre-defined for this problem.

    """
    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, rosenbrock,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([1.0] * self.num_variables)]



def schaffer6(phenome):
    """The bare-bones Schaffer function 6."""
    sum_of_squares = phenome[0] ** 2 + phenome[1] ** 2
    result = math.sin(math.sqrt(sum_of_squares)) ** 2 - 0.5
    result /= (1.0 + 0.001 * sum_of_squares) ** 2
    result += 0.5
    return result



class Schaffer6(TestProblem):
    """Schaffer's test problem 6.

    This problem is radially symmetric. Thus it does not possess a discrete
    set of local optima. It was defined for two dimensions in
    [Schaffer1989]_. The global optimum is the origin and the search space
    is :math:`[-100, 100] \\times [-100, 100]`.

    References
    ----------
    .. [Schaffer1989] Schaffer, J. David; Caruana, Richard A.; Eshelman,
        Larry J.; Das, Rajarshi (1989). A study of control parameters
        affecting online performance of genetic algorithms for function
        optimization. In: Proceedings of the third international
        conference on genetic algorithms, pp. 51-60, Morgan Kaufmann.

    """
    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 2
        self._min_bounds = [-100.0, -100.0]
        self._max_bounds = [100.0, 100.0]
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, schaffer6,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



def schaffer7(phenome):
    """The bare-bones Schaffer function 7."""
    sum_of_squares = phenome[0] ** 2 + phenome[1] ** 2
    result = sum_of_squares ** 0.25
    result *= math.sin(50.0 * sum_of_squares ** 0.1) ** 2 + 1.0
    return result



class Schaffer7(TestProblem):
    """Schaffer's test problem 7.

    This problem is radially symmetric. Thus it does not possess a discrete
    set of local optima. It was defined for two dimensions in
    [Schaffer1989]_. The global optimum is the origin and the search space
    is :math:`[-100, 100] \\times [-100, 100]`.

    """
    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 2
        self._min_bounds = [-100.0, -100.0]
        self._max_bounds = [100.0, 100.0]
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, schaffer7,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]



def schwefel(phenome):
    """The bare-bones Schwefel function."""
    ret = 0.0
    sin = math.sin
    sqrt = math.sqrt
    for x in phenome:
        ret -= x * sin(sqrt(abs(x)))
    return ret



class Schwefel(TestProblem):
    """Schwefel's test problem.

    Note that bound constraints are required for the global optimum to
    exist. :math:`[-500, 500]` for each variable is the default here.
    Then the problem has :math:`7^n` local optima.

    """
    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        self._min_bounds = [-500.0] * num_variables
        self._max_bounds = [500.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, schwefel,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([420.96874635998199] * self.num_variables)]


    def get_locally_optimal_solutions(self, max_number=None):
        """Return the locally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        # shortcuts
        sqrt = math.sqrt
        sin = math.sin
        cos = math.cos
        copysign = math.copysign
        min_bounds = self.min_bounds
        max_bounds = self.max_bounds

        def first_derivative(x):
            sqrt_x = sqrt(x)
            return -sin(sqrt_x) - 0.5 * sqrt_x * cos(sqrt_x)

        def second_derivative(x):
            sqrt_x = sqrt(x)
            return 0.25 * sin(sqrt_x) - (0.75 * cos(sqrt_x)) / sqrt_x

        assert len(min_bounds) == len(max_bounds)
        min_bound = min(min_bounds)
        max_bound = max(max_bounds)
        max_root_pos = int(copysign(sqrt(abs(max_bound)) / math.pi, max_bound))
        min_root_pos = int(copysign(sqrt(abs(min_bound)) / math.pi, min_bound))
        minima_positions = []
        for pos in range(min_root_pos, max_root_pos):
            if pos % 2 == 0:
                # initial estimate (center between two zeros)
                old_x = ((pos * math.pi) ** 2 + ((pos + 1.0) * math.pi) ** 2) * 0.5
                new_x = old_x - first_derivative(old_x) / second_derivative(old_x)
                # newton's method
                counter = 0
                while abs(new_x - old_x) > 1e-12 and counter < 20:
                    old_x = new_x
                    new_x = old_x - first_derivative(old_x) / second_derivative(old_x)
                    counter += 1
                minima_positions.append(copysign(new_x, pos))
        # filter feasible positions in each dimension
        positions_in_dimensions = []
        for dim in range(self.num_variables):
            positions_in_this_dim = []
            for pos in minima_positions:
                if pos >= min_bounds[dim] and pos <= max_bounds[dim]:
                    positions_in_this_dim.append(pos)
            positions_in_dimensions.append(positions_in_this_dim)
        optima = []
        if max_number is None:
            max_number = float("inf")
        # build cross product of all dimensions
        for position in itertools.product(*positions_in_dimensions):
            optima.append(Individual(list(position)))
            if len(optima) >= max_number:
                break
        return optima



def six_hump_camelback(phenome):
    """The bare-bones six-hump camelback function."""
    x1 = phenome[0]
    x2 = phenome[1]
    part1 = (4.0 - 2.1 * x1 ** 2 + (x1 ** 4) / 3.0) * x1 ** 2
    return 4.0 * (part1 + x1 * x2 + (-4.0 + 4.0 * x2 ** 2) * x2 ** 2)



class SixHumpCamelback(TestProblem):
    """The so-called six-hump camelback test problem.

    No bound constraints are pre-defined for this problem. Possible choices
    including all the optima are :math:`[-1.9, 1.9] \\times [-1.1, 1.1]` or
    larger rectangles.

    """
    def __init__(self, phenome_preprocessor=None, **kwargs):
        self.num_variables = 2
        preprocessor = SequenceChecker(self.num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, six_hump_camelback,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the two global optima."""
        optima = []
        phenomes = [[0.089842007286237896, -0.71265640548186626],
                    [-0.089842007286237896, 0.71265640548186626]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max(1, max_number)]
        return optima


    def get_locally_optimal_solutions(self, max_number=None):
        """Return the locally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        optima = self.get_optimal_solutions()
        phenomes = [[-1.7036067132900241, 0.7960835697790869],
                    [1.7036067132900241, -0.7960835697790869],
                    [-1.6071047491815618, -0.56865145239564607],
                    [1.6071047607120053, 0.56865145738534051]]
        for phenome in phenomes:
            optima.append(Individual(phenome))
        if max_number is not None:
            optima = optima[:max(1, max_number)]
        return optima



def sphere(phenome):
    """The bare-bones sphere function."""
    return sum(x ** 2 for x in phenome)



class Sphere(TestProblem):
    """The sphere problem.

    Possibly the most simple unimodal problem. No bound constraints are
    pre-defined.

    """
    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, sphere,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]


    get_locally_optimal_solutions = get_optimal_solutions



def vincent(phenome):
    """The bare-bones Vincent function."""
    sin = math.sin
    log = math.log
    ret = 0.0
    for x in phenome:
        ret += sin(10.0 * log(x))
    return -ret / len(phenome)



class Vincent(TestProblem):
    """Vincent's test problem.

    All variables have lower and upper bounds of 0.25 and 10, respectively.
    The problem has :math:`6^n` global optima.

    """
    def __init__(self, num_variables=5, phenome_preprocessor=None, **kwargs):
        self.num_variables = num_variables
        self._min_bounds = [0.25] * num_variables
        self._max_bounds = [10.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.bound_constraints_checker = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, vincent,
                             phenome_preprocessor=self.bound_constraints_checker,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @property
    def min_bounds(self):
        return self._min_bounds


    @min_bounds.setter
    def min_bounds(self, bounds):
        self._min_bounds = bounds
        self.bound_constraints_checker.min_bounds = bounds


    @property
    def max_bounds(self):
        return self._max_bounds


    @max_bounds.setter
    def max_bounds(self, bounds):
        self._max_bounds = bounds
        self.bound_constraints_checker.max_bounds = bounds


    def get_optimal_solutions(self, max_number=None):
        """Return the optimal solutions.

        All local optima are global optima in this problem.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        # shortcuts
        ceil = math.ceil
        floor = math.floor
        log = math.log
        min_bounds = self.min_bounds
        max_bounds = self.max_bounds
        # find out how many minima there are in the feasible space
        # first transform limits to "log"-space
        transformed_min_bounds = [10.0 * log(min_bound) for min_bound in min_bounds]
        transformed_max_bounds = [10.0 * log(max_bound) for max_bound in max_bounds]
        # find the multiples of 2*pi that are closest to the bounds
        min_counters = []
        for transformed_min_bound in transformed_min_bounds:
            min_counters.append(ceil((transformed_min_bound - HALF_PI) / TWO_PI))
        max_counters = []
        for transformed_max_bound in transformed_max_bounds:
            max_counters.append(floor((transformed_max_bound + HALF_PI) / TWO_PI))
        # optima are at every multiple in between
        ranges = []
        for min_counter, max_counter in zip(min_counters, max_counters):
            ranges.append(list(range(int(min_counter), int(max_counter) + 1)))
        optima = []
        if max_number is None:
            max_number = float("inf")
        # build cross product of all dimensions
        for position in itertools.product(*ranges):
            opt = Individual()
            # carry out inverse transformation
            opt.phenome = [math.exp((TWO_PI * pos + HALF_PI) / 10.0) for pos in position]
            optima.append(opt)
            if len(optima) >= max_number:
                break
        return optima


    get_locally_optimal_solutions = get_optimal_solutions



class WeierstrassFunction:
    """A configurable Weierstrass function."""

    def __init__(self, a=0.5, b=3.0, k_max=20):
        self.a = a
        self.b = b
        self.k_max = k_max


    def __call__(self, phenome):
        """Evaluate the function."""
        n = len(phenome)
        a = self.a
        b = self.b
        k_max = self.k_max
        sum1 = 0.0
        cos = math.cos
        for i in range(n):
            for k in range(k_max + 1):
                sum1 += a ** k * cos(TWO_PI * b ** k * (phenome[i] + 0.5))
        sum2 = 0.0
        for k in range(k_max + 1):
            sum2 += a ** k * cos(TWO_PI * b ** k * 0.5)
        return sum1 - n * sum2



class Weierstrass(TestProblem):
    """Weierstrass' test problem.

    No bound constraints are pre-defined for this problem.

    """
    def __init__(self, num_variables=10,
                 a=0.5,
                 b=3.0,
                 k_max=20,
                 phenome_preprocessor=None,
                 **kwargs):
        self.num_variables = num_variables
        preprocessor = SequenceChecker(num_variables,
                                       previous_preprocessor=phenome_preprocessor)
        TestProblem.__init__(self, WeierstrassFunction(a, b, k_max),
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    def get_optimal_solutions(self, max_number=None):
        """Return the global optimum."""
        return [Individual([0.0] * self.num_variables)]
