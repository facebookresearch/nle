# MiniHack Environments

## Navigation

Navigation tasks require the agent to move towards the goal (staircase down). The action space in navigation tasks is moving towards 8 compas directions (+ pickup and apply in some cases, kick).

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
- `MiniHack-Empty-10x10-v0`
- `MiniHack-Empty-Random-10x10-v0`

In this environment, the agent is placed in an empty room. In the randomised versions of this task, the starting point of the agent are randomly sample for each episode. In the fixed version of the task, the agent always starts from the top-left corder and the goal is in the bottum-right corner. A small penalty is subtracted for the number of steps to reach the goal.
