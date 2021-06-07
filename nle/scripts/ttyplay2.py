from nle.nethack.actions import _ACTIONS_DICT
import nle.scripts.ttyplay as ttyplay

if __name__ == "__main__":
    ttyplay.ACTIONS = _ACTIONS_DICT
    ttyplay.main()
