from nle.env import tasks as nle_tasks
from nle.minihack.envs import room


ENVS = dict(
    # NLE tasks
    staircase=nle_tasks.NetHackStaircase,
    score=nle_tasks.NetHackScore,
    pet=nle_tasks.NetHackStaircasePet,
    oracle=nle_tasks.NetHackOracle,
    gold=nle_tasks.NetHackGold,
    eat=nle_tasks.NetHackEat,
    scout=nle_tasks.NetHackScout,
    # MiniHack Room
    small_room=room.MiniHackRoom5x5,
    small_room_random=room.MiniHackRoom5x5Random,
    small_room_dark=room.MiniHackRoom5x5Dark,
    small_room_monster=room.MiniHackRoom5x5Monster,
    small_room_trap=room.MiniHackRoom5x5Trap,
    big_room=room.MiniHackRoom15x15,
    big_room_random=room.MiniHackRoom15x15Random,
    big_room_dark=room.MiniHackRoom15x15Dark,
    big_room_monster=room.MiniHackRoom15x15Monster,
    big_room_monster_trap=room.MiniHackRoom15x15MonsterTrap,
)
