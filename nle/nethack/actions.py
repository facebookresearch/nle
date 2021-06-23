# Copyright (c) Facebook, Inc. and its affiliates.
import enum


def M(c):
    if isinstance(c, str):
        c = ord(c)
    return 0x80 | c


def C(c):
    if isinstance(c, str):
        c = ord(c)
    return 0x1F & c


class TextCharacters(enum.IntEnum):
    PLUS = ord("+")
    MINUS = ord("-")
    SPACE = ord(" ")
    APOS = ord("'")
    QUOTE = ord('"')
    NUM_0 = ord("0")
    NUM_1 = ord("1")
    NUM_2 = ord("2")
    NUM_3 = ord("3")
    NUM_4 = ord("4")
    NUM_5 = ord("5")
    NUM_6 = ord("6")
    NUM_7 = ord("7")
    NUM_8 = ord("8")
    NUM_9 = ord("9")
    DOLLAR = ord("$")  # TODO: Add SEE* actions instead (see PR #177).


class CompassCardinalDirection(enum.IntEnum):
    N = ord("k")
    E = ord("l")
    S = ord("j")
    W = ord("h")


class CompassIntercardinalDirection(enum.IntEnum):
    NE = ord("u")
    SE = ord("n")
    SW = ord("b")
    NW = ord("y")


# Merge the above.
CompassDirection = enum.IntEnum(
    "CompassDirection",
    {
        **CompassCardinalDirection.__members__,
        **CompassIntercardinalDirection.__members__,
    },
)


class CompassCardinalDirectionLonger(enum.IntEnum):
    N = ord("K")
    E = ord("L")
    S = ord("J")
    W = ord("H")


class CompassIntercardinalDirectionLonger(enum.IntEnum):
    NE = ord("U")
    SE = ord("N")
    SW = ord("B")
    NW = ord("Y")


CompassDirectionLonger = enum.IntEnum(
    "CompassDirectionLonger",
    {
        **CompassCardinalDirectionLonger.__members__,
        **CompassIntercardinalDirectionLonger.__members__,
    },
)


class MiscDirection(enum.IntEnum):
    UP = ord("<")  # go up a staircase
    DOWN = ord(">")  # go down a staircase
    WAIT = ord(".")  # rest one move while doing nothing / apply to self


class MiscAction(enum.IntEnum):
    MORE = ord("\r")  # read the next message


class UnsafeActions(enum.IntEnum):
    # currently these result in an error or undesirable behaviour
    HELP = ord("?")  # give a help message
    PREVMSG = C("p")  # view recent game messages


class Command(enum.IntEnum):
    EXTCMD = ord("#")  # perform an extended command
    EXTLIST = M("?")  # list all extended commands
    ADJUST = M("a")  # adjust inventory letters
    ANNOTATE = M("A")  # name current level
    APPLY = ord("a")  # apply (use) a tool (pick-axe, key, lamp...)
    ATTRIBUTES = C("x")  # show your attributes
    AUTOPICKUP = ord("@")  # toggle the pickup option on/off
    CALL = ord("C")  # call (name) something
    CAST = ord("Z")  # zap (cast) a spell
    CHAT = M("c")  # talk to someone
    CLOSE = ord("c")  # close a door
    CONDUCT = M("C")  # list voluntary challenges you have maintained
    DIP = M("d")  # dip an object into something
    DROP = ord("d")  # drop an item
    DROPTYPE = ord("D")  # drop specific item types
    EAT = ord("e")  # eat something
    ESC = C("[")  # escape from the current query/action
    ENGRAVE = ord("E")  # engrave writing on the floor
    ENHANCE = M("e")  # advance or check weapon and spell skills
    FIRE = ord("f")  # fire ammunition from quiver
    FIGHT = ord("F")  # Prefix: force fight even if you don't see a monster
    FORCE = M("f")  # force a lock
    GLANCE = ord(";")  # show what type of thing a map symbol corresponds to
    HISTORY = ord("V")  # show long version and game history
    INVENTORY = ord("i")  # show your inventory
    INVENTTYPE = ord("I")  # inventory specific item types
    INVOKE = M("i")  # invoke an object's special powers
    JUMP = M("j")  # jump to another location
    KICK = C("d")  # kick something
    KNOWN = ord("\\")  # show what object types have been discovered
    KNOWNCLASS = ord("`")  # show discovered types for one class of objects
    LOOK = ord(":")  # look at what is here
    LOOT = M("l")  # loot a box on the floor
    MONSTER = M("m")  # use monster's special ability
    MOVE = ord("m")  # Prefix: move without picking up objects/fighting
    MOVEFAR = ord("M")  # Prefix: run without picking up objects/fighting
    OFFER = M("o")  # offer a sacrifice to the gods
    OPEN = ord("o")  # open a door
    OPTIONS = ord("O")  # show option settings, possibly change them
    OVERVIEW = C("o")  # show a summary of the explored dungeon
    PAY = ord("p")  # pay your shopping bill
    PICKUP = ord(",")  # pick up things at the current location
    PRAY = M("p")  # pray to the gods for help
    PUTON = ord("P")  # put on an accessory (ring, amulet, etc)
    QUAFF = ord("q")  # quaff (drink) something
    QUIT = M("q")  # exit without saving current game
    QUIVER = ord("Q")  # select ammunition for quiver
    READ = ord("r")  # read a scroll or spellbook
    REDRAW = C("r")  # redraw screen
    REMOVE = ord("R")  # remove an accessory (ring, amulet, etc)
    RIDE = M("R")  # mount or dismount a saddled steed
    RUB = M("r")  # rub a lamp or a stone
    RUSH = ord("g")  # Prefix: rush until something interesting is seen
    RUSH2 = ord("G")  # Prefix: rush until something interesting is seen
    SAVE = ord("S")  # save the game and exit
    SEARCH = ord("s")  # search for traps and secret doors
    SEEALL = ord("*")  # show all equipment in use
    SEETRAP = ord("^")  # show the type of adjacent trap
    SIT = M("s")  # sit down
    SWAP = ord("x")  # swap wielded and secondary weapons
    TAKEOFF = ord("T")  # take off one piece of armor
    TAKEOFFALL = ord("A")  # remove all armor
    TELEPORT = C("t")  # teleport around the level
    THROW = ord("t")  # throw something
    TIP = M("T")  # empty a container
    TRAVEL = ord("_")  # travel to a specific location on the map
    TURN = M("t")  # turn undead away
    TWOWEAPON = ord("X")  # toggle two-weapon combat
    UNTRAP = M("u")  # untrap something
    VERSION = M("v")  # list compile time options for this version of NetHack
    VERSIONSHORT = ord("v")  # show version
    WEAR = ord("W")  # wear a piece of armor
    WHATDOES = ord("&")  # tell what a command does
    WHATIS = ord("/")  # show what type of thing a symbol corresponds to
    WIELD = ord("w")  # wield (put in use) a weapon
    WIPE = M("w")  # wipe off your face
    ZAP = ord("z")  # zap a wand


ACTIONS = tuple(
    list(CompassDirection)
    + list(CompassDirectionLonger)
    + list(MiscDirection)
    + list(MiscAction)
    + list(Command)
    + list(TextCharacters)
)

NON_RL_ACTIONS = (
    Command.ANNOTATE,  # Could potentially be useful.
    Command.AUTOPICKUP,
    Command.CONDUCT,  # Could potentially be useful.
    Command.EXTCMD,  # Potentially useful for some wizard actions.
    Command.EXTLIST,
    Command.GLANCE,
    Command.HISTORY,
    Command.KNOWN,  # Could potentially be useful.
    Command.KNOWNCLASS,  # Could potentially be useful.
    Command.OPTIONS,
    Command.OVERVIEW,  # Could potentially be useful.
    Command.TELEPORT,
    Command.QUIT,
    Command.REDRAW,
    Command.SAVE,
    Command.SEEALL,  # Could potentially be useful.
    Command.TRAVEL,  # Could potentially be useful.
    Command.VERSION,
    Command.WHATDOES,
    Command.WHATIS,
)

_USEFUL_ACTIONS = list(ACTIONS)
for action in NON_RL_ACTIONS + tuple(TextCharacters):
    _USEFUL_ACTIONS.remove(action)
_USEFUL_ACTIONS.append(TextCharacters.SPACE)
_USEFUL_ACTIONS.append(TextCharacters.DOLLAR)
USEFUL_ACTIONS = tuple(_USEFUL_ACTIONS)
del _USEFUL_ACTIONS

_ACTIONS_DICT = {}
for enum_class in [
    TextCharacters,
    CompassDirection,
    CompassDirectionLonger,
    MiscDirection,
    MiscAction,
    Command,
]:
    for action in enum_class:
        _ACTIONS_DICT[action.value] = "%s.%s" % (enum_class.__name__, action.name)


def action_id_to_type(action):
    return _ACTIONS_DICT[action]
