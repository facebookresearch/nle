# MiniHack Environments

## Navigation

Navigation tasks require the agent to move towards the goal (staircase down). The action space in navigation tasks is moving towards 8 compas directions is by default. In several cases, a subset of the following actions might also be available: `kick`, `open`, `search`, `pickup`, `apply`.
### Empty


```
.....
.@...
.....
.....
...>.
```

Existing configurations:
- `MiniHack-Empty-5x5-v0`
- `MiniHack-Empty-Random-5x5-v0`
- `MiniHack-Empty-10x10-v0`
- `MiniHack-Empty-Random-10x10-v0`
- `MiniHack-Empty-15x15-v0`
- `MiniHack-Empty-Random-15x15-v0`

In this environment, the agent is placed in an empty room. In the randomised versions of this task, the starting point of the agent are randomly sample for each episode. In the fixed version of the task, the agent always starts from the top-left corder and the goal is in the bottum-right corner. 

The small version of this environment can be used to verify/debug implementations of RL algorithms, whereas the bigger versions can be used for initial experimentations methods designed for of sparse rewards settings. 

### Corridor

```

                                           ---------
   ------------    ------                  |.......+
   |..........|    |.....######         ###.......<|
   |...........####|....|     ## -----###  |.......|
   ------------   #|.....      ##|....#    ---------
                  #--.-.-       #|...|
                  # ##          #|...|
                  #####        ##|...|
               ####             #-...|
            ---.-------         #--|--
            |.........|        #######     -----        ----+------- ---
            |.........|      ###  ##      #|...|     ###-...@..........|
            |.........-#######     ########....|   ###  |..............|
            |.........|                   #|...|   #    |..............|
            |.........|                   #|...|  ##    |.......>......|
            |.........|                   #----- ##     ----------------
            -----------                   ########

```

Existing configurations:
- `MiniHack-Corridor-R2-v0`
- `MiniHack-Corridor-R3-v0`
- `MiniHack-Corridor-R5-v0`
- `MiniHack-Corridor-R8-v0`
- `MiniHack-Corridor-R10-v0`

## Skill Tasks

TODO

## MiniGrid Tasks

These tasks are ported from [MiniGrid](https://github.com/maximecb/gym-minigrid). If you use them in your experiments, please cite the original work.

### MultiRoom

```
                             -------
                             |.....|
                             |.....|
                             |.....|
                             |.....|
                             |.....|
                             |.@...|
                             |.....|
                          ----+-----
                          |.....|
                          |.....|---
                          |.....|..|---------
                          |.....|--|..||....|
                          |.....+..+..||.>..|
                          |.....|..|..||....|
                          |.....|..|..|---+-----
                          |.....|..|..|........|
                          ------|..|..+........|
                                ------|........|
                                      ----------
```

Existing configurations:
- `MiniHack-MultiRoom-N2-S4-v0` (two small rooms)
- `MiniHack-MultiRoom-N4-S5-v0` (four rooms)
- `MiniHack-MultiRoom-N6-v0` (six rooms)

### MonsterMultiRoom

Existing configurations:
- `MiniHack-MonsterMultiRoom-N2-S4-v0` (two small rooms)
- `MiniHack-MonsterMultiRoom-N4-S5-v0` (four rooms)
- `MiniHack-MonsterMultiRoom-N6-v0` (six rooms)

### LavaCrossing

```
                                  -----------
                                  |@....}.}.|
                                  |}}}.}}}}}|
                                  |.}.....}.|
                                  |}}}}}}.}}|
                                  |.}...}.}.|
                                  |.}...}.}.|
                                  |.}...}...|
                                  |.}...}.}.|
                                  |.}...}.}>|
                                  -----------
```

Existing configurations:
- `MiniHack-LavaCrossingS9N1-v0`
- `MiniHack-LavaCrossingS9N2-v0`
- `MiniHack-LavaCrossingS9N3-v0`
- `MiniHack-LavaCrossingS11N5-v0`


### SimpleCrossing

```
                                  ---------
                                  |@|...|.|
                                  |.|...|.|
                                  |.|.....|
                                  |.....|.|
                                  |.|...|.|
                                  ------|.---
                                        |...|
                                        ---.|
                                          |>|
                                          ---
```
- `MiniHack-SimpleCrossingS9N1-v0`
- `MiniHack-SimpleCrossingS9N2-v0`
- `MiniHack-SimpleCrossingS9N3-v0`
- `MiniHack-SimpleCrossingS11N5-v0`
