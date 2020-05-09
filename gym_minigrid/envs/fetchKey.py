from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class FetchKeyEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random color key
    named using English text strings
    """

    def __init__(
        self,
        size=8,
        numObjs=3,
    ):
        self.numObjs = numObjs

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)


        types = ['key']
        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:

            objColor = self._rand_elem(COLOR_NAMES)
            obj = Key(objColor)

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = '%s %s' % (self.targetColor, self.targetType)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.targetColor and \
               self.carrying.type == self.targetType:
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info


class FetchKeyEnv5x5N5(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=5)


class FetchKeyEnv6x6N5(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=5)


class FetchKeyEnv8x8N5(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=5)


class FetchKeyEnv16x16N5(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=16, numObjs=5)




register(
    id='MiniGrid-fetchKey-5x5-N5-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv5x5N5'
)

register(
    id='MiniGrid-fetchKey-6x6-N5-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv6x6N5'
)

register(
    id='MiniGrid-fetchKey-8x8-N5-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv8x8N5'
)

register(
    id='MiniGrid-fetchKey-16x16-N5-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv16x16N5'
)

#############################################################

class FetchKeyEnv5x5N3(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=3)


class FetchKeyEnv6x6N3(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=3)


class FetchKeyEnv8x8N3(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=3)


class FetchKeyEnv16x16N3(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=16, numObjs=3)




register(
    id='MiniGrid-fetchKey-5x5-N3-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv5x5N3'
)

register(
    id='MiniGrid-fetchKey-6x6-N3-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv6x6N3'
)

register(
    id='MiniGrid-fetchKey-8x8-N3-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv8x8N3'
)

register(
    id='MiniGrid-fetchKey-16x16-N3-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv16x16N3'
)


####################################################

class FetchKeyEnv5x5N2(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=2)


class FetchKeyEnv6x6N2(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=2)


class FetchKeyEnv8x8N2(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=2)


class FetchKeyEnv16x16N2(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=16, numObjs=2)




register(
    id='MiniGrid-fetchKey-5x5-N2-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv5x5N2'
)

register(
    id='MiniGrid-fetchKey-6x6-N2-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv6x6N2'
)

register(
    id='MiniGrid-fetchKey-8x8-N2-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv8x8N2'
)

register(
    id='MiniGrid-fetchKey-16x16-N2-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv16x16N2'
)

##############################
class FetchKeyEnv3x3N2(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=3, numObjs=2)


class FetchKeyEnv16x16N1(FetchKeyEnv):
    def __init__(self):
        super().__init__(size=3, numObjs=1)

register(
    id='MiniGrid-fetchKey-3x3-N2-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv3x3N2'
)

register(
    id='MiniGrid-fetchKey-3x3-N1-v0',
    entry_point='gym_minigrid.envs:FetchKeyEnv3x3N1'
)






