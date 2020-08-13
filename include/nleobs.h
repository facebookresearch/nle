
#ifndef NLEOBS_H
#define NLEOBS_H

#define NLE_BLSTATS_SIZE 25

typedef struct nle_observation {
    int action;
    int done;
    short *glyphs;           /* Size ROWNO * (COLNO - 1) */
    unsigned char *chars;    /* Size ROWNO * (COLNO - 1) */
    unsigned char *colors;   /* Size ROWNO * (COLNO - 1) */
    unsigned char *specials; /* Size ROWNO * (COLNO - 1) */
    unsigned char *message;  /* Size 256 */
    long *blstats;           /* NLE_BLSTATS_SIZE */
    int *program_state;      /* Size 5 */
    int *internal;           /* Size 5 */
} nle_obs;

#endif /* NLEOBS_H */
