import importlib

from hpg.environments.flipbit import FlipBit

from hpg.environments.mazes import FourRooms
from hpg.environments.mazes import EmptyRoom


environments = {'FlipBit': FlipBit,
                'EmptyRoom': EmptyRoom,
                'FourRooms': FourRooms}


ale_python_interface = importlib.util.find_spec('ale_python_interface')
if ale_python_interface is not None:
    from hpg.environments.pacman import MsPacman
    environments['MsPacman'] = MsPacman


mujoco_py = importlib.util.find_spec('mujoco_py')
if mujoco_py is not None:
    from hpg.environments.robotics import FetchPush
    environments['FetchPush'] = FetchPush