/*
 * Wrapper for dlopen.
 */

#ifndef NLEDL_H
#define NLEDL_H

#include <stdio.h>

#include "nleobs.h"

typedef struct nledl_ctx {
    void *dlhandle;
    void *nle_ctx;
    void (*step)(void *, nle_obs *);
    FILE *outfile;
} nle_ctx_t;

nle_ctx_t *nle_start(nle_obs *);
nle_ctx_t *nle_step(nle_ctx_t *, nle_obs *);

void nle_reset(nle_ctx_t *, nle_obs *);
void nle_end(nle_ctx_t *);

#endif /* NLEDL_H */
