import abc

# this defines the interface to the robot controller


class RobotControllerInterface(metaclass=abc.ABCMeta):

    def __init__(self, dim_output):
        assert isinstance(dim_output, int) and dim_output > 0, \
            "Expected 'dim_output' to be a positive integer, but it is not."
        self._dim_output = dim_output

        # # set random seed
        # if seed is None:
        #     seed = np.random.randint(1, 10000)
        # np.random.seed(seed)

    @abc.abstractmethod
    def get_input(self, observation, reference):
        """
        Given the current observation, calculate the desired input to system
        """


"""
consider making this interface class
own the object of the participant controller
this may limit any weirdness of participants messing with the parent class
"""
