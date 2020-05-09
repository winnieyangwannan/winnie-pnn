from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class FetchObjEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch either a yellow key or a blue ball
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

        types = ['key', 'ball','box', 'goal','door']
        objs = []

        # For each object to be generated
        objType = self._rand_elem(types[0:3])
        if objType == 'key':
            obj = Key('yellow')
        elif objType == 'ball':
            obj = Ball('yellow')
        elif objType == 'box':
            obj = Box('yellow')

        self.place_obj(obj)
        objs.append(obj)

        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            #objColor = self._rand_elem(COLOR_NAMES)

            if objType == 'key':
                obj = Key('yellow')
            elif objType == 'ball':
                obj = Ball('yellow')
            elif objType == 'goal':
                obj = Goal('yellow')
            elif objType == 'box':
                obj = Box('yellow')
            elif objType == 'door':
                obj = Door('yellow')

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[0]
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


class FetchObjEnv5x5N2(FetchObjEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=2)


class FetchObjEnv6x6N2(FetchObjEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=2)


class FetchObjEnv8x8N2(FetchObjEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=2)


class FetchObjEnv16x16N2(FetchObjEnv):
    def __init__(self):
        super().__init__(size=16, numObjs=2)


register(
    id='MiniGrid-FetchObj-5x5-N2-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv5x5N2'
)

register(
    id='MiniGrid-FetchObj-6x6-N2-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv6x6N2'
)

register(
    id='MiniGrid-FetchObj-8x8-N2-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv8x8N2'
)

register(
    id='MiniGrid-FetchObj-16x16-N2-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv16x16N2'
)

##########################################################################
class FetchObjEnv5x5N3(FetchObjEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=3)


class FetchObjEnv6x6N3(FetchObjEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=3)


class FetchObjEnv8x8N3(FetchObjEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=3)


class FetchObjEnv16x16N3(FetchObjEnv):
    def __init__(self):
        super().__init__(size=16, numObjs=3)



register(
    id='MiniGrid-FetchObj-5x5-N3-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv5x5N3'
)

register(
    id='MiniGrid-FetchObj-6x6-N3-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv6x6N3'
)

register(
    id='MiniGrid-FetchObj-8x8-N3-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv8x8N3'
)

register(
    id='MiniGrid-FetchObj-16x16-N3-v0',
    entry_point='gym_minigrid.envs:FetchObjEnv16x16N3'
)



