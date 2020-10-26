
#ifndef NLEOBS_H
#define NLEOBS_H

#define NLE_MESSAGE_SIZE 256
#define NLE_BLSTATS_SIZE 25
#define NLE_PROGRAM_STATE_SIZE 6
#define NLE_INTERNAL_SIZE 7
#define NLE_INVENTORY_SIZE 55
#define NLE_INVENTORY_STR_LENGTH 80
#define NLE_GLYPH_STR_LENGTH 80

typedef struct nle_observation {
    int action;
    int done;
    char in_normal_game;     /* Bool indicating if other obs are set. */
    short *glyphs;           /* Size ROWNO * (COLNO - 1) */
    unsigned char *chars;    /* Size ROWNO * (COLNO - 1) */
    unsigned char *colors;   /* Size ROWNO * (COLNO - 1) */
    unsigned char *specials; /* Size ROWNO * (COLNO - 1) */
    long *blstats;           /* NLE_BLSTATS_SIZE */
    unsigned char *message;  /* Size NLE_MESSAGE_SIZE */
    int *program_state;      /* NLE_PROGRAM_STATE_SIZE */
    int *internal;           /* NLE_INTERNAL_SIZE */
    short *inv_glyphs;       /* NLE_INVENTORY_SIZE */
    unsigned char
        *inv_strs; /* NLE_INVENTORY_SIZE * NLE_INVENTORY_STR_LENGTH */
    unsigned char *inv_letters;  /* NLE_INVENTORY_SIZE */
    unsigned char *inv_oclasses; /* NLE_INVENTORY_SIZE */
    unsigned char *glyph_strs;  /* Size ROWNO * (COLNO - 1) * NLE_GLYPH_STR_LENGTH */
} nle_obs;

typedef struct {
    unsigned long seeds[2]; /* core, disp */
    char reseed; /* boolean: use NetHack's anti-TAS reseed mechanism? */
} nle_seeds_init_t;

#endif /* NLEOBS_H */
