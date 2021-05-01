from nle.minihack import MiniHackNavigation, LevelGenerator
import gym


class MiniGridHack(MiniHackNavigation):
    def __init__(self, *args, **kwargs):
        import gym_minigrid  # noqa: F401

        self._minigrid_env = gym.make(kwargs["env_name"])
        self.num_mon = kwargs.pop("num_mon", 0)

        des_file = self.get_env_desc()
        super().__init__(des_file=des_file)

    def get_env_map(self, env):
        door_pos = []
        goal_pos = None
        empty_strs = 0
        empty_str = True
        env_map = []

        for j in range(env.grid.height):
            str = ""
            for i in range(env.width):
                c = env.grid.get(i, j)
                if c is None:
                    str += "."
                    continue
                empty_str = False
                if c.type == "wall":
                    str += "|"
                elif c.type == "door":
                    str += "+"
                    door_pos.append((i, j - empty_strs))
                elif c.type == "floor":
                    str += "."
                elif c.type == "lava":
                    str += "L"
                elif c.type == "goal":
                    goal_pos = (i, j - empty_strs)
                    str += "."
                elif c.type == "player":
                    str += "."
            if not empty_str and j < env.grid.height - 1:
                if set(str) != {"."}:
                    str = str.replace(".", " ", str.index("|"))
                    inv = str[::-1]
                    str = inv.replace(".", " ", inv.index("|"))[::-1]
                    env_map.append(str)
            elif empty_str:
                empty_strs += 1

        start_pos = (int(env.agent_pos[0]), int(env.agent_pos[1]) - empty_strs)
        env_map = "\n".join(env_map)

        return env_map, start_pos, goal_pos, door_pos

    def get_env_desc(self):
        self._minigrid_env.reset()
        env = self._minigrid_env

        map, start_pos, goal_pos, door_pos = self.get_env_map(env)

        lev_gen = LevelGenerator(map=map)

        lev_gen.add_stair_down(goal_pos)
        lev_gen.add_stair_up(start_pos)
        for d in door_pos:
            lev_gen.add_door("closed", d)
        lev_gen.wallify()

        for _ in range(self.num_mon):
            lev_gen.add_monster()

        return lev_gen.get_des()

    def reset(self):
        des_file = self.get_env_desc()
        self.update(des_file)
        return super().reset()
