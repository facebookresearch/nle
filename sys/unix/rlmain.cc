#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <termios.h>
#include <unistd.h>

extern "C" {
#include "hack.h"
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
        for (int i = 0; i < 23; ++i) {
            std::cout << obs->blstats[i] << " ";
        }
        std::cout << std::endl;
        read(STDIN_FILENO, &obs->action, 1);
        if (obs->action == 'r')
            nle_reset(nle, obs, nullptr, nullptr);
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

    for (int i = 0; i < 15; ++i) {
        randplay(nle, obs);
        nle_reset(nle, obs, nullptr, nullptr);
    }
}

int
main(int argc, char **argv)
{
    struct termios old, tty;
    tcgetattr((int) STDIN_FILENO, &old);
    tty = old;
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);

    nle_obs obs{};
    constexpr int dungeon_size = ROWNO * (COLNO - 1);
    short glyphs[dungeon_size];
    obs.glyphs = &glyphs[0];

    unsigned char chars[dungeon_size];
    obs.chars = &chars[0];

    unsigned char colors[dungeon_size];
    obs.colors = &colors[0];

    unsigned char specials[dungeon_size];
    obs.specials = &specials[0];

    unsigned char message[256];
    obs.message = &message[0];

    long blstats[NLE_BLSTATS_SIZE];
    obs.blstats = &blstats[0];

    int program_state[NLE_PROGRAM_STATE_SIZE];
    obs.program_state = &program_state[0];

    int internal[NLE_INTERNAL_SIZE];
    obs.internal = &internal[0];

    std::unique_ptr<FILE, int (*)(FILE *)> ttyrec(fopen("nle.ttyrec", "a"),
                                                  fclose);

    nle_ctx_t *nle = nle_start("libnethack.so", &obs, ttyrec.get(), nullptr);
    if (argc > 1 && argv[1][0] == 'r') {
        randgame(nle, &obs);
    } else {
        play(nle, &obs);
        nle_reset(nle, &obs, nullptr, nullptr);
        play(nle, &obs);
    }
    nle_end(nle);

    tcsetattr(STDIN_FILENO, TCSANOW, &old);
}
