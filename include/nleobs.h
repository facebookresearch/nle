
#ifndef NLEOBS_H
#define NLEOBS_H

#define NLE_MESSAGE_SIZE 256
#define NLE_BLSTATS_SIZE 25
#define NLE_PROGRAM_STATE_SIZE 6
#define NLE_INTERNAL_SIZE 9
#define NLE_MISC_SIZE 3
#define NLE_INVENTORY_SIZE 55
#define NLE_INVENTORY_STR_LENGTH 80
#define NLE_SCREEN_DESCRIPTION_LENGTH 80
#define NLE_TERM_CO 80
#define NLE_TERM_LI 24

/* #define NLE_ALLOW_SEEDING 1 */

typedef struct nle_observation {
    int action;
    int done;
    char in_normal_game;     /* Bool indicating if other obs are set. */
    int how_done;            /* If game is really_done, how it ended. */
    short *glyphs;           /* Size ROWNO * (COLNO - 1) */
    unsigned char *chars;    /* Size ROWNO * (COLNO - 1) */
    unsigned char *colors;   /* Size ROWNO * (COLNO - 1) */
    unsigned char *specials; /* Size ROWNO * (COLNO - 1) */
    long *blstats;           /* Size NLE_BLSTATS_SIZE */
    unsigned char *message;  /* Size NLE_MESSAGE_SIZE */
    int *program_state;      /* Size NLE_PROGRAM_STATE_SIZE */
    int *internal;           /* Size NLE_INTERNAL_SIZE */
    short *inv_glyphs;       /* Size NLE_INVENTORY_SIZE */
    unsigned char
        *inv_strs; /* Size NLE_INVENTORY_SIZE * NLE_INVENTORY_STR_LENGTH */
    unsigned char *inv_letters;         /* Size NLE_INVENTORY_SIZE */
    unsigned char *inv_oclasses;        /* Size NLE_INVENTORY_SIZE */
    unsigned char *screen_descriptions; /* Size ROWNO * (COLNO - 1) *
                                           NLE_SCREEN_DESCRIPTION_LENGTH */
    unsigned char *tty_chars;           /* Size NLE_TERM_LI * NLE_TERM_CO */
    signed char *tty_colors;            /* Size NLE_TERM_LI * NLE_TERM_CO */
    unsigned char *tty_cursor;          /* Size 2 */
    int *misc;                          /* Size NLE_MISC_SIZE */
} nle_obs;

typedef struct {
#ifdef NLE_ALLOW_SEEDING
    unsigned long seeds[2]; /* core, disp */
    char reseed; /* boolean: use NetHack's anti-TAS reseed mechanism? */
#endif
} nle_seeds_init_t;

#endif /* NLEOBS_H */
