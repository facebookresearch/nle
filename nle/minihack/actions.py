# Copyright (c) Facebook, Inc. and its affiliates.

import string
import enum
from nle.nethack import CompassDirection, CompassDirectionLonger

ACTION_STR_DICT = {
    "N": "move north",
    "E": "move east",
    "S": "move south",
    "W": "move west",
    "NE": "move northeast",
    "SE": "move southeast",
    "SW": "move southwest",
    "NW": "move northwest",
    "UP": "up",
    "DOWN": "down",
    "WAIT": "wait",
    "MORE": "more",
    "EXTCMD": "extended command",  # NON RL
    "EXTLIST": "extended command list",  # NON RL
    "ADJUST": "adjust",
    "ANNOTATE": "annotate",  # NON RL
    "APPLY": "apply",
    "ATTRIBUTES": "show attributes",
    "AUTOPICKUP": "autopickup",  # NON RL
    "CALL": "call",
    "CAST": "cast",
    "CHAT": "chat",
    "CLOSE": "close",
    "CONDUCT": "conduct",  # NON RL
    "DIP": "dip",
    "DROP": "drop",
    "DROPTYPE": "drop type",
    "EAT": "eat",
    "ESC": "escape",
    "ENGRAVE": "engrave",
    "ENHANCE": "enhance",
    "FIRE": "fire",
    "FIGHT": "fight",
    "FORCE": "force",
    "GLANCE": "glance",  # NON RL
    "HELP": "help",
    "HISTORY": "history",  # NON RL
    "INVENTORY": "show inventory",
    "INVENTTYPE": "inventory type",
    "INVOKE": "invoke",
    "JUMP": "jump",
    "KICK": "kick",
    "KNOWN": "known",  # NON RL
    "KNOWNCLASS": "known class",  # NOW RL
    "LOOK": "look",
    "LOOT": "loot",
    "MONSTER": "use monster's ability",
    "MOVE": "move",
    "MOVEFAR": "move far",
    "OFFER": "offer",
    "OPEN": "open",
    "OPTIONS": "options",  # NON RL
    "OVERVIEW": "overview",  # NON RL
    "PAY": "pay",
    "PICKUP": "pick up",
    "PRAY": "pray",
    "PREVMSG": "previous message",  # NON RL
    "PUTON": "put on",
    "QUAFF": "quaff",
    "QUIT": "quit",  # NON RL
    "QUIVER": "quiver",
    "READ": "read",
    "REDRAW": "redraw",  # NON RL
    "REMOVE": "remove",
    "RIDE": "ride",
    "RUB": "rub",
    "RUSH": "rush",
    "SAVE": "save",  # NON RL
    "SEARCH": "search",
    "SEEALL": "see all",  # NON RL
    "SEETRAP": "see trap type",
    "SIT": "sit",
    "SWAP": "swap weapons",
    "TAKEOFF": "take off",
    "TAKEOFFALL": "take off all",
    "TELEPORT": "teleport",
    "THROW": "throw",
    "TIP": "tip",
    "TRAVEL": "travel",  # NON RL
    "TURN": "turn",
    "TWOWEAPON": "wield two weapon",
    "UNTRAP": "untrap",
    "VERSION": "version",  # NON RL
    "VERSIONSHORT": "version short",  # NON RL?
    "WEAR": "wear",
    "WHATDOES": "what does",  # NON RL
    "WHATIS": "what is",  # NON RL
    "WIELD": "wield",
    "WIPE": "wipe",
    "ZAP": "zap",
}


InventorySelection = enum.IntEnum(
    "InventorySelection",
    {k: ord(k) for k in string.ascii_letters},
)


def action_to_str(action, inventory=None):
    if isinstance(action, InventorySelection) and inventory is not None:
        return inventory[action.name]

    assert (
        action.name in ACTION_STR_DICT
    ), f"Action {action} cannot be mapped to a string"
    return ACTION_STR_DICT[action.name]


def action_to_name(action, inventory=None):
    if isinstance(action, InventorySelection) and inventory is not None:
        return inventory[action.name]

    if isinstance(action, CompassDirection) or isinstance(
        action, CompassDirectionLonger
    ):
        return ACTION_STR_DICT[action.name]

    return action.name.lower()
