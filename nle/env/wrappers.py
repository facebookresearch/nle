import ctypes
import weakref

import _ctypes
import gym


class CTypesAPIWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.dll = None
        self.dll_ref = None

    def get_dll(self):
        if self.dll is None:
            self.dll = ctypes.CDLL(self.env.unwrapped.env.dlpath)
            weakref.finalize(self.dll, _ctypes.dlclose, self.dll._handle)
            self.dll_ref = weakref.ref(self.dll)
        return self.dll

    def reset(self, **kwargs):
        if self.dll_ref is not None:
            self.dll = None
            if self.dll_ref() is not None:
                raise RuntimeError("References to Nethack DLL present at reset")
            self.dll_ref = None
        return self.env.reset(**kwargs)

    def seed(self, core=None, disp=None, reseed=False):
        return self.env.seed(core, disp, reseed)
