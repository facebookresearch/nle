# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np


PROGRAM_STATE_FIELDS = [
    "Gameover",
    "Panicking",
    "Exiting",
    "InMoveloop",
    "InImpossible",
]

STATUS_FIELDS = [
    "TITLE",
    "STR",
    "DX",
    "CO",
    "IN",
    "WI",
    "CH",  # 1..6
    "ALIGN",
    "SCORE",
    "CAP",
    "GOLD",
    "ENE",
    "ENEMAX",  # 7..12
    "XP",
    "AC",
    "HD",
    "TIME",
    "HUNGER",
    "HP",
    "HPMAX",
    "LEVELDESC",  # 13..20
    "EXP",
]

CONDITION_FIELDS = [
    "STONE",
    "SLIME",
    "STRNGL",
    "FOODPOIS",
    "TERMILL",
    "BLIND",
    "DEAF",
    "STUN",
    "CONF",
    "HALLU",
    "LEV",
    "FLY",
    "RIDE",
]

BLSTATS_FIELDS = [
    "CursX",
    "CursY",
    "StrengthPercentage",
    "Strength",
    "Dexterity",
    "Constitution",
    "Intelligence",
    "Wisdom",
    "Charisma",
    "Score",
    "Hitpoints",
    "MaxHitpoints",
    "Depth",
    "Gold",
    "Energy",
    "MaxEnergy",
    "ArmorClass",
    "MonsterLevel",
    "ExperienceLevel",
    "ExperiencePoints",
    "Time",
    "HungerState",
    "CarryingCapacity",
]


INVENTORYITEM_FIELDS = ["Glyph", "Str", "Letter", "ObjectClass", "ObjectClassName"]

MENUITEM_FIELDS = ["Glyph", "Selector", "Gselector", "Str", "Selected"]

WIN_MESSAGE = 1  # Technically dynamic. Practically constant.

SEEDS_FIELDS = ["Core", "Disp"]


def fb_ndarray_to_np(fb_ndarray):
    result = fb_ndarray.DataAsNumpy()
    result = result.view(np.typeDict[fb_ndarray.Dtype()])
    result = result.reshape(fb_ndarray.ShapeAsNumpy().tolist())
    return result


def print_message(message):
    fb_windows = [message.Windows(i) for i in range(message.WindowsLength())]

    for fb_window in fb_windows:
        window = {}
        window["strings"] = [
            fb_window.Strings(i) for i in range(fb_window.StringsLength())
        ]
        window["items"] = []
        fb_items = [fb_window.MenuItems(i) for i in range(fb_window.MenuItemsLength())]
        for fb_item in fb_items:
            window["items"].append(
                {field: getattr(fb_item, field)() for field in MENUITEM_FIELDS}
            )
        window["type"] = fb_window.Type()
        print("window", window)

    seeds = {k: getattr(message.Seeds(), k)() for k in SEEDS_FIELDS}
    print("seeds", seeds)

    if message.NotRunning():
        return "done"

    program_state = message.ProgramState()
    program_state_dict = {
        field: getattr(program_state, field)() for field in PROGRAM_STATE_FIELDS
    }
    print("program_state", program_state_dict)

    if not program_state.InMoveloop():
        print("Game not in move loop.")
        return

    obs = message.Observation()
    status = obs.Status()
    status_dict = {
        field: getattr(status, field)().decode("utf-8") for field in STATUS_FIELDS
    }
    condition = status.Condition()
    condition_dict = {field: getattr(condition, field)() for field in CONDITION_FIELDS}
    print("status", status_dict)
    print("condition", condition_dict)

    blstats = message.Blstats()
    blstats_dict = {field: getattr(blstats, field)() for field in BLSTATS_FIELDS}
    print("blstats", blstats_dict)

    inventory = [obs.Inventory(i) for i in range(obs.InventoryLength())]
    items = [
        {field: getattr(item, field)() for field in INVENTORYITEM_FIELDS}
        for item in inventory
    ]
    print("inv_items", items)
    for item in items:
        print(chr(item["Letter"]), item["Str"])

    internal = message.Internal()
    print("deepest_lev_reached", internal.DeepestLevReached())
    if internal.KillerName():
        print("killer_name", internal.KillerName())
    call_stack = [internal.CallStack(i) for i in range(internal.CallStackLength())]
    if internal.Xwaitforspace():
        print("Waiting for space/return")
    print("call_stack", call_stack)
    if internal.StairsDown():
        print("On downward stairs")

    message_win = message.Windows(WIN_MESSAGE)
    messages = [message_win.Strings(i) for i in range(message_win.StringsLength())]
    if messages:
        print("Messages:")
        for m in messages:
            print(m)

    chars = fb_ndarray_to_np(obs.Chars())
    colors = fb_ndarray_to_np(obs.Colors())
    rows, cols = chars.shape
    nh_HE = "\033[0m"
    BRIGHT = 8
    for r in range(rows):
        if False:  # no colors.
            print(chars[r].tobytes().decode("utf-8"))
            continue
        for c in range(cols):
            # cf. termcap.c.
            start_color = "\033[%d" % bool(colors[r][c] & BRIGHT)
            start_color += ";3%d" % (colors[r][c] & ~BRIGHT)
            start_color += "m"

            print(start_color + chr(chars[r][c]), end=nh_HE)
        print("")
