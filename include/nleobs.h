
#ifndef NLEOBS_H
#define NLEOBS_H

#define NLE_BLSTATS_SIZE 25
#define NLE_PROGRAM_STATE_SIZE 6
#define NLE_INTERNAL_SIZE 7

typedef struct nle_observation {
    int action;
    int done;
    char in_normal_game;     /* Bool indicating if other obs are set. */
    short *glyphs;           /* Size ROWNO * (COLNO - 1) */
    unsigned char *chars;    /* Size ROWNO * (COLNO - 1) */
    unsigned char *colors;   /* Size ROWNO * (COLNO - 1) */
    unsigned char *specials; /* Size ROWNO * (COLNO - 1) */
    unsigned char *message;  /* Size 256 */
    long *blstats;           /* NLE_BLSTATS_SIZE */
    int *program_state;      /* NLE_PROGRAM_STATE_SIZE */
    int *internal;           /* NLE_INTERNAL_SIZE */
} nle_obs;

typedef struct {
    unsigned long seeds[2]; /* core, disp */
    char reseed; /* boolean: use NetHack's anti-TAS reseed mechanism? */
} nle_seeds_init_t;

#endif /* NLEOBS_H */
