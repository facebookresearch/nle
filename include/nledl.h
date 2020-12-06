/*
 * Wrapper for dlopen.
 */

#ifndef NLEDL_H
#define NLEDL_H

#include <stdio.h>

#include "nleobs.h"

/* TODO: Don't call this nle_ctx_t as well. */
typedef struct nledl_ctx nle_ctx_t;

nle_ctx_t *nle_start(const char *, nle_obs *, FILE *, nle_seeds_init_t *, int shared);
nle_ctx_t *nle_step(nle_ctx_t *, nle_obs *);

void nle_reset(nle_ctx_t *, nle_obs *, FILE *, nle_seeds_init_t *);
void nle_end(nle_ctx_t *);

void nle_set_seed(nle_ctx_t *, unsigned long, unsigned long, char);
void nle_get_seed(nle_ctx_t *, unsigned long *, unsigned long *, char *);

int nle_supports_shared(void);

#endif /* NLEDL_H */
