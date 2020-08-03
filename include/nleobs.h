
#ifndef NLEOBS_H
#define NLEOBS_H

typedef struct nle_observation {
    int action;
    int done;
    short *glyphs;  /* Size ROWNO * (COLNO - 1) */
    char *chars;    /* Size ROWNO * (COLNO - 1) */
    char *colors;   /* Size ROWNO * (COLNO - 1) */
    char *specials; /* Size ROWNO * (COLNO - 1) */
} nle_obs;

#endif /* NLEOBS_H */
