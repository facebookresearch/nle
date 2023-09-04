/*
 * Wrapper for dlopen.
 */

#ifndef NLEDL_H
#define NLEDL_H

#include <stdio.h>

#include "nleobs.h"

typedef struct nledl_ctx {
    char dlpath[1024];
    void *dlhandle;
    void *nle_ctx;
    void *(*step)(void *, nle_obs *);
    FILE *ttyrec;
} nledl_ctx;

nledl_ctx *nle_start(const char *, nle_obs *, FILE *, nle_seeds_init_t *,
                     nle_settings *);
nledl_ctx *nle_step(nledl_ctx *, nle_obs *);

void nle_reset(nledl_ctx *, nle_obs *, FILE *, nle_seeds_init_t *,
               nle_settings *);
void nle_end(nledl_ctx *);

void nle_set_seed(nledl_ctx *, unsigned long, unsigned long, char);
void nle_get_seed(nledl_ctx *, unsigned long *, unsigned long *, char *);

int nle_save(nledl_ctx *);

#endif /* NLEDL_H */
