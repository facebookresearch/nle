import ctypes

import _ctypes

import nle.nethack


class mvital(ctypes.Structure):
    """From decl.h."""

    _fields_ = [
        ("born", ctypes.c_ubyte),
        ("died", ctypes.c_ubyte),
        ("mvflags", ctypes.c_ubyte),
    ]


mvitals = mvital * nle.nethack.NUMMONS


def main():
    e = nle.nethack.Nethack()

    dl = ctypes.CDLL(e.dlpath)

    nroom = ctypes.c_int.in_dll(dl, "nroom")
    print(nroom)
    e.reset()
    print(nroom)

    mv = mvitals.in_dll(dl, "mvitals")

    for i, m in enumerate(mv):
        if m.mvflags == 0:
            continue
        pm = nle.nethack.permonst(i)
        print(pm.mname, m.mvflags)

    for i in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(i)
        if pm.mname == "minotaur":
            print(mv[i].died, pm.mname, "has/have died")
            break

    # Important! Otherwise, the next reset() will core dump.
    _ctypes.dlclose(dl._handle)

    e.reset()
    e.reset()
    e.close()


main()
