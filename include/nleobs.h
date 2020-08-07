
#ifndef NLEOBS_H
#define NLEOBS_H

typedef struct nle_observation {
    int action;
    int done;
    short *glyphs;           /* Size ROWNO * (COLNO - 1) */
    unsigned char *chars;    /* Size ROWNO * (COLNO - 1) */
    unsigned char *colors;   /* Size ROWNO * (COLNO - 1) */
    unsigned char *specials; /* Size ROWNO * (COLNO - 1) */
    unsigned char *message;  /* Size 256 */
    long *blstats;           /* Size 23 */
    int *program_state;      /* Size 5 */
    int *internal;           /* Size 5 */
} nle_obs;

#endif /* NLEOBS_H */
