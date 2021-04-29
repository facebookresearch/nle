import os
import gym

from tempfile import NamedTemporaryFile
from nle.minihack import MiniHackNavigation


class MiniGridHackMultiroom(MiniHackNavigation):
    def __init__(self, *args, **kwargs):
        import gym_minigrid  # noqa: F401

        self._minigrid_env = gym.make(kwargs["env_name"])

        env_desc = self.get_env_desc()
        f = NamedTemporaryFile(delete=False, suffix=".des")
        with open(f.name, "w") as tmp:
            tmp.write("\n".join(env_desc))
        f.close()
        super().__init__(des_file=f.name)
        os.unlink(f.name)

    def get_env_desc(self):
        self._minigrid_env.reset()
        env = self._minigrid_env

        env_desc = [
            "MAZE: \"mylevel\", ' '",
            "FLAGS: premapped",
            "INIT_MAP: solidfill, ' '",
            "GEOMETRY: center, center",
            "MAP",
        ]

        door_positions = []
        player_pos = (env.agent_pos[0], env.agent_pos[1])
        goal_position = None
        empty_strs = 0
        empty_str = True
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
                if c.type == "door":
                    str += "+"
                    door_positions.append((i, j - empty_strs))
                    continue
                if c.type == "floor":
                    str += "."
                    continue
                if c.type == "goal":
                    goal_position = (i, j - empty_strs)
                    str += "."
                    continue
                if c.type == "player":
                    str += "."
                    continue
            if not empty_str and j < env.grid.height - 1:
                if set(str) != {"."}:
                    str = str.replace(".", " ", str.index("|"))
                    inv = str[::-1]
                    str = inv.replace(".", " ", inv.index("|"))[::-1]
                    env_desc.append(str)
            elif empty_str:
                empty_strs += 1
        env_desc.append("ENDMAP")
        env_desc.extend([f"DOOR: closed, {d}" for d in door_positions])
        env_desc.append(f"STAIR: {goal_position}, down")
        ppx = player_pos[0]
        ppy = player_pos[1]
        env_desc.append(
            f"BRANCH:" f"{ppx, ppy-empty_strs, ppx, ppy-empty_strs}" f",(0,0,0,0)"
        )
        env_desc.append("WALLIFY")
        return env_desc

    def get_env_desc_slow(self):

        self._minigrid_env.reset()
        env = self._minigrid_env

        env_desc = [
            "MAZE: \"mylevel\", ' '",
            "FLAGS: premapped",
            "INIT_MAP: solidfill, ' '",
            "GEOMETRY: center, center",
            "MAP",
        ]

        mg_level = env.__str__().split("\n")
        mg_level = [el for el in mg_level if el.strip() != ""]
        mg_level[0] = mg_level[0].replace("WG", "|")
        mg_level[-1] = mg_level[-1].replace("WG", "|")
        mg_level[0] = mg_level[0].replace(" ", "|")
        mg_level[-1] = mg_level[-1].replace(" ", "|")

        player_chars = [">>", "<<", "VV", "^^"]
        COLS = ["R", "G", "B", "P", "Y", "G"]
        door_chars = [f"D{c}" for c in COLS]

        door_strs = []
        for i in range(1, len(mg_level) - 1):
            for pc in player_chars:
                if pc in mg_level[i]:
                    pcx = (mg_level[i].index(pc) + 1) // 2
                    player_str = f"BRANCH:{pcx, i, pcx, i},(0,0,0,0)"
                    break

            if "GG" in mg_level[i]:
                stair_str = f"STAIR: {((mg_level[i].index('GG') + 1) // 2, i)}, down"
            for dc in door_chars:
                if dc in mg_level[i]:
                    door_strs.append(
                        f"DOOR: closed, {(mg_level[i].index(dc) + 1) // 2, i}"
                    )

            for pc in player_chars:
                mg_level[i] = mg_level[i].replace(pc, ".")
            for dc in door_chars:
                mg_level[i] = mg_level[i].replace(dc, "+")

            mg_level[i] = mg_level[i].replace("  ", ".")
            mg_level[i] = mg_level[i].replace("WG", "|")
            mg_level[i] = mg_level[i].replace("GG", ".")
            mg_level[i] = mg_level[i].replace(".", " ", mg_level[i].index("|"))
            inv = mg_level[i][::-1]
            mg_level[i] = inv.replace(".", " ", inv.index("|"))[::-1]

        env_desc.extend(mg_level)
        env_desc.append("ENDMAP")
        env_desc.extend(door_strs)
        env_desc.append(stair_str)
        env_desc.append(player_str)
        return env_desc

    def reset(self):
        env_desc = self.get_env_desc()
        f = NamedTemporaryFile(delete=False, suffix=".des")
        with open(f.name, "w") as tmp:
            tmp.write("\n".join(env_desc))
        f.close()
        self.update(f.name)
        os.unlink(f.name)
        return super().reset()
