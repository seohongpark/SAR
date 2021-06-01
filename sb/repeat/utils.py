import numpy as np

from gym import spaces


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    elif isinstance(action_space, spaces.Tuple):
        sum = 0
        for space in action_space.spaces:
            sum += get_action_dim(space)
        return sum
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
