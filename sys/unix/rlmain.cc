
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <termios.h>
#include <unistd.h>

extern "C" {
#include "hack.h"
}

/*
extern "C" {
#include "dlb.h"
}
*/

extern "C" {
#include "nledl.h"
}

void
play(nle_ctx_t *nle, nle_obs *obs)
{
    char i;
    while (!obs->done) {
        for (int r = 0; r < ROWNO; ++r) {
            for (int c = 0; c < COLNO - 1; ++c)
                std::cout << obs->chars[r * (COLNO - 1) + c];
            std::cout << std::endl;
        }
        read(STDIN_FILENO, &obs->action, 1);
        nle = nle_step(nle, obs);
    }
}

void
randplay(nle_ctx_t *nle, nle_obs *obs)
{
    int actions[] = {
        13, 107, 108, 106, 104, 117, 110, 98, 121,
        75, 76,  74,  72,  85,  78,  66,  89,
    };
    size_t n = sizeof(actions) / sizeof(actions[0]);

    while (!obs->done) {
        obs->action = actions[rand() % n];
        nle = nle_step(nle, obs);
    }
}

void
randgame(nle_ctx_t *nle, nle_obs *obs)
{
    obs->action = 'y';
    nle_step(nle, obs);
    nle_step(nle, obs);
    obs->action = '\n';
    nle_step(nle, obs);

    for (int i = 0; i < 5; ++i) {
        randplay(nle, obs);
        nle_reset(nle, obs);
    }
}

int
main(int argc, char **argv)
{
    /*std::cerr << "short break before beginning: ";
      int i;
      read(STDIN_FILENO, &i, 1);
    */

    struct termios old, tty;
    tcgetattr((int) STDIN_FILENO, &old);
    tty = old;
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);

    nle_obs obs;
    constexpr int dungeon_size = ROWNO * (COLNO - 1);
    short glyphs[dungeon_size];
    obs.glyphs = &glyphs[0];
    char chars[dungeon_size];
    obs.chars = &chars[0];

    nle_ctx_t *nle = nle_start(&obs);
    randgame(nle, &obs);
    play(nle, &obs);
    nle_reset(nle, &obs);
    play(nle, &obs);
    nle_end(nle);

    /*
    std::cerr << "short break before end: ";
    read(STDIN_FILENO, &i, 1);
    */
    tcsetattr(STDIN_FILENO, TCSANOW, &old);
}
