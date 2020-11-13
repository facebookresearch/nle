# Architecture

This document is aims to clarify the architecture of NLE, to assist those
looking to contribute to fundamental development.  

## Preface

The NLE Repo is a fork of [nethack/nethack](https://github.com/NetHack/NetHack/releases/tag/NetHack-3.6.6_PostRelease),
the 3.6.6 post release (commit: `36ee184`), modified to allow for integration of
a new interface to the game suitable for RL. The architecture can be thought of
as operating at through various layers connecting the low C files that execute
the game, up to the openai `gym` environment that just feeds back observations.
Below we examine the these layers from lowest to highest:


### Layer 0: “Original” NetHack Game Logic

- **part of `libnethack.so`**
- **key directories & files** [`src/`](https://github.com/facebookresearch/nle/blob/master/src),
[`include/`](https://github.com/facebookresearch/nle/blob/master/include),
[`dat/`](https://github.com/facebookresearch/nle/blob/master/dat),
[`src/allmain.c`](https://github.com/facebookresearch/nle/blob/master/src/allmain.c)


* The original nethack game is written purely in C, found mainly in `src` and
`include`
* As a C program, it relies heavily on global variables shared throughout the
file to keep track of the internal state of the game, rendering the gamely
utterly un-threadsafe.
* The game logic itself starts in  `allmain.c`  where the game is started with
[`void moveloop(bool resuming)`](https://github.com/facebookresearch/nle/blob/master/src/allmain.c#L23).
This function never returns, instead it runs an infinite loop
[(line 83)](https://github.com/facebookresearch/nle/blob/master/src/allmain.c#L83),
and may be interrupted either by the player or completing the game.

### Layer 1: Interacting with NetHack Game Logic: Windows and Window Procs

- **part of `libnethack.so`**
- **key directories & files** [`win/rl`](https://github.com/facebookresearch/nle/blob/master/win/rl),
[`win/tty`](https://github.com/facebookresearch/nle/blob/master/win/tty),
[`src/windows.c`](https://github.com/facebookresearch/nle/blob/master/src/windows.c),
[`include/winprocs.h`](https://github.com/facebookresearch/nle/blob/master/include/winprocs.h)


* Periodically the game may need input from the player, or need to return
something to be displayed to the player. To allow a variety of interfaces to be
exposed to the player (ie tty, mobile, atari etc), a developer must implement
and provide a number functions to the game, as a standard contract.
* The full api is described in
[`doc/window.doc`](https://github.com/facebookresearch/nle/blob/master/doc/window.doc),
and is defined in `include/winprocs.h`. The object to be provided to the game is the
`struct window_procs`, which is (largely) a struct of pointers to functions.
* Once a developer has implemented this struct, it can be added to the chosen
`window_procs` in `src/windows.c`, via some macros. Our `window_procs` struct
is `extern`'d there under the name `rl_procs`.
* The implementation of `rl_procs` is found in
[`win/rl/winrl.cc`](https://github.com/facebookresearch/nle/blob/master/win/rl/winrl.cc).  
In this C++ file we implement a `class NetHackRL` that will contain all our
functions, and then we declare the struct `rl_procs` to point to the appropriate
methods.
* Our implementation of the `window_procs` wrap heavily around existing tty
`window_procs` found in
[`win/tty/wintty.c`](https://github.com/facebookresearch/nle/blob/master/win/tty/wintty.c)
but allows us to control the capture of information and when to send it to the
user.  

### Layer 2: Yielding from NetHack Game Logic: Context Switching and the “nethack-stack”

- **part of `libnethack.so`**
- **key directories & files** [`src/nle.c`](https://github.com/facebookresearch/nle/blob/master/src/nle.c),
[`include/nle.h`](https://github.com/facebookresearch/nle/blob/master/include/nle.h),
[`include/nleobs.h`](https://github.com/facebookresearch/nle/blob/master/include/nleobs.h)


* In our case the user “playing” the game is not a terminal, but ultimately a
python program that we want to be able to run and execute at the same time. In
fact we want to be able to step through nethack and our python code (roughly)
simultaneously.  For performance reasons we chose to do this all in the same
thread, stepping through each program in turn. This will require context
switching for each program.   
* Since our context switching needs to be in C++,  we will use
`<fcontext/fcontext.h>`, and we will write a “layer” that wraps around the
nethack game, allocating a stack on the heap to play it (the “nethack-stack“),
and swapping into it periodically when we need to step through the game. To
allow the game to yield back to the caller, we also implement a yielding
function which we will need to insert into the relevant `rl_procs` described
above.  
* Our “layer” is a group of functions found in `src/nle.c`, exposed in
`include/nle.h`.  Alongside these functions we add the yielding function
`nle_yield` which will be used to cede control from the nethack-stack, back to
user.:
    * `void * nle_yield(void *not_done)`
        * This function yields from the nethack-stack back to user. When it
        resumes, it returns data that can be cast to `nle_obs *` , and contains
        an action.
        * It is called in only one window_proc, the `getch_method()`  - used to
        get characters/actions from the user
    * `nle_ctx_t * nle_start(nle_obs *obs, FILE *ttyrec, nle_seeds_init_t *seeds_init)`
        * This function allocates a block of memory on the heap, and turns it
        into a stack.
        * On this stack it switches context and executes `main_loop()`, a thin
        wrapper around `sys/unix/unixmain.c`’s main function - effectively the
        same as calling "`nethack`" from yout command line.
    * `nle_ctx_t * nle_step(nle_ctx_t *nle, nle_obs *obs)`
        * This function takes an (pointer to) an observation (which includes an
        action) and exposes it to the nethack-stack context, and ceding control
        to that nethack-stack from the user
    * `void nle_end(nle_ctx_t * nle)`
        * clean up memory, destroy the nethack-stack, pray nethack in Layer 0
        hasnt leaked memory on the heap.
* Other useful objects in this layer include:
    * `nle_ctx_t from include/nle.h`
        * a struct that contains the stack type and the two contexts
        (`return_context`, `generator_context`) which are the user and
        nethack-stack contexts respectively.
    * `nle_obs from include/nleobs.h`
        * a struct that contains the observations transferred between the two
        stacks/contexts

### Layer 3: Dynamic Library Loading: Resetting the Game

- **library:** libnethackdl.a,
- **key directories & files:** [`sys/unix/nledl.c`](https://github.com/facebookresearch/nle/blob/master/sys/unix/nledl.c),
[`include/nledl.h`](https://github.com/facebookresearch/nle/blob/master/include/nledl.h),
[`sys/unix/rlmain.cc`](https://github.com/facebookresearch/nle/blob/master/sys/unix/rlmain.cc)


* Now that we have a means of playing the game in `libnethack.so`, we would
like to reset the game when it finishes.  However, as mentioned above, nethack
ends a game abruptly by sending a signal to its `moveloop` leaving the libraries
global state as is.  
* Since there is no way to reset this manually, we will need to reset the
library by loading it ‘from scratch’, using `dlopen` and `dlclose`, so we can play
again.
* This problem is solved by the `libnethackdl.a` library, which provides a simple
layer of indirection to the usage of `libnethack.so`. This library is very small,
simply consisting of thin wrappers around `dlopen` and `dlclose`, found in
`sys/unix/nledl.c` and its header `include/nledl.h` . The wrappers effectively
mirror the api of `nle.h` but also take a string for the name of the library to
be opened.
* Now we can simply use the interface in nledl.h for all interacting with
nethack, instead of `nle.h`, and in fact we can test this with the file
`sys/unix/rlmain.cc` which produces an executable that can be tested after build.

### Layer 4: Binding to Python: Exposing the Game

- **libraries:** `_pynethack.cpython-37.*`
- **key directories & files:**[`pynethack.cc`](https://github.com/facebookresearch/nle/blob/master/win/rl/pynethack.cc),
[`src/monst.c`](https://github.com/facebookresearch/nle/blob/master/src/monst.c),
[`src/objects.c`](https://github.com/facebookresearch/nle/blob/master/src/objects.c),
[`src/decl.c`](https://github.com/facebookresearch/nle/blob/master/src/decl.c),
[`src/drawing.c`](https://github.com/facebookresearch/nle/blob/master/src/drawing.c)


* Now that we have a means of playing the game, we need to bind our commands to
Python
* We can define a class in `win/rl/pynethack.cc`, which we will expose in python
using PyBind11 bindings. In particular we define `class Nethack`, which has
methods like `step` and `reset` which wrap around out `nledl.h` methods like
`nle_step` and `nle_start` / `nle_reset`.
* At the bottom of this file we then define the module `_pynethack` which will
contain that class as well as a number of global variables we wish to expose.
These include items found in `src/monst.c`, `src/decl.c`, `src/drawing.c`,
`src/objects.c`

### Layer 5: Calling from Python: Playing A Copy of the Game

- **libraries:** [`nle`](https://github.com/facebookresearch/nle/blob/master/nle)
- **key directories & files:** [`nle/nethack/*.py`](https://github.com/facebookresearch/nle/blob/master/nle/nethack),
[`nle/env/*.py`](https://github.com/facebookresearch/nle/blob/master/nle/env)


* The `class nle.nethack.Nethack` is our Python interface to the game. It does
has two important roles:
    - It implements a “copy” of the `libnethack.so` library into a temporary
    directory used to play nethack.
    -  It then creates an instance of `_pynethack.Nethack`, using the path to
    that library (and a pointer to `ttyrec` being recorded).
* The reason for the copy is relatively simple: the `libnethack` game stores all
its data as global library data. If we want to be able to play multiple
instances of the game simultaneously, we need to open different copies of the
same library. It turns out python will only open each library once and then
return the same object each time. Currently we get round this by having each
instance of nle.nethack.Nethack copying the library (only ~1MB) to its own
temporary `HACKDIR`, and using that.
* So we now have an instance of `nle.nethack.Nethack` which wraps around
`_pynethack.Nethack` and also defines some important enums for the game (action
types etc) in `nle.nethack.actions*`. All that remains is to implement the API
required by OpenAI gym environment. This base implementation can be found in
`nle.env.base`   as `class NLE(gym.Env)`. which implements various definitions to
do with the action and state space, and also reward functions and rendering.
* Finally we have derived tasks in nle.env.task that allow users to expand and
define their own tasks, for instance staircase etc.
