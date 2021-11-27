# Copyright (c) Facebook, Inc. and its affiliates.
import ctypes
import weakref

import _ctypes
import gym


class CTypesAPIWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.dll = None

    def get_dll(self):
        if self.dll is None:
            self.dll = ctypes.CDLL(self.env.unwrapped.env.dlpath)
            weakref.finalize(self.dll, _ctypes.dlclose, self.dll._handle)
        return self.dll

    def reset(self, **kwargs):
        if self.dll is not None:
            ref = weakref.ref(self.dll)
            self.dll = None
            if ref() is not None:
                raise RuntimeError("References to Nethack DLL present at reset")
        return self.env.reset(**kwargs)

    def seed(self, core=None, disp=None, reseed=False):
        return self.env.seed(core, disp, reseed)
