from nle.minihack import MiniHackNavigation
from nle.minihack.level_generator import KeyRoomGenerator
from gym.envs import registration
from nle.nethack import Command
from nle import nethack

MOVE_ACTIONS = tuple(nethack.CompassDirection)
APPLY_ACTIONS = tuple(list(MOVE_ACTIONS) + [Command.PICKUP, Command.APPLY])


class MiniHackKeyDoor(MiniHackNavigation):
    """Environment for "key and door" task."""

    def __init__(self, *args, des_file, **kwargs):
        kwargs["options"] = kwargs.pop("options", list(nethack.NETHACKOPTIONS))
        kwargs["options"].append("!autopickup")
        kwargs["character"] = kwargs.pop("charachter", "rog-hum-cha-mal")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        kwargs["actions"] = kwargs.pop("actions", APPLY_ACTIONS)
        super().__init__(*args, des_file=des_file, **kwargs)

    def step(self, action: int):
        # If apply action is chosen
        if self._actions[action] == Command.APPLY:
            key_key = self.key_in_inventory("key")
            # if key is in the inventory
            if key_key is not None:
                # Check if there is a closed door nearby
                dir_key = self.get_direction_obj("closed door")
                if dir_key is not None:
                    # Perform the following NetHack steps
                    self.env.step(Command.APPLY)  # press apply
                    self.env.step(ord(key_key))  # choose key from the inv
                    self.env.step(dir_key)  # select the door's direction
                    obs, done = self.env.step(ord("y"))  # press y
                    obs, done = self._perform_known_steps(obs, done, exceptions=True)
                    # Make sure the door is open
                    while True:
                        obs, done = self.env.step(dir_key)
                        obs, done = self._perform_known_steps(
                            obs, done, exceptions=True
                        )
                        if self.get_direction_obj("closed door", obs) is None:
                            break

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info


class MiniHackKeyRoom(MiniHackKeyDoor):
    def __init__(self, *args, room_size, subroom_size, lit, **kwargs):
        lev_gen = KeyRoomGenerator(room_size, subroom_size, lit)
        des_file = lev_gen.get_des()
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackKeyRoom5x5(MiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, room_size=5, subroom_size=2, lit=True, **kwargs)


class MiniHackKeyRoom5x5Dark(MiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, room_size=5, subroom_size=2, lit=False, **kwargs)


class MiniHackKeyRoom15x15(MiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        super().__init__(*args, room_size=15, subroom_size=5, lit=True, **kwargs)


class MiniHackKeyRoom15x15Dark(MiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        super().__init__(*args, room_size=15, subroom_size=5, lit=False, **kwargs)


# KeyRoom
registration.register(
    id="MiniHack-KeyRoom-S5-v0",
    entry_point="nle.minihack.envs.keyroom:MiniHackKeyRoom5x5",
)
registration.register(
    id="MiniHack-KeyRoom-S15-v0",
    entry_point="nle.minihack.envs.keyroom:MiniHackKeyRoom15x15",
)
registration.register(
    id="MiniHack-KeyRoom-Unlit-S5-v0",
    entry_point="nle.minihack.envs.keyroom:MiniHackKeyRoom5x5Dark",
)
registration.register(
    id="MiniHack-KeyRoom-Unlit-S15-v0",
    entry_point="nle.minihack.envs.keyroom:MiniHackKeyRoom15x15Dark",
)
