
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nledl.h"

#if defined(__linux__) && defined(__x86_64__)
#define HASSHARED
#endif

void* nleshared_open(const char *dlpath);
void nleshared_close(void* handle);
void nleshared_reset(void* handle);
void* nleshared_sym(void* handle, const char* symname);

typedef struct nledl_ctx {
    void* shared;
    char dlpath[1024];
    void *dlhandle;
    void *nle_ctx;
    void *(*start)(nle_obs *, FILE *, nle_seeds_init_t *);
    void *(*step)(void *, nle_obs *);
    void (*end)(void *);
    FILE *ttyrec;
} nle_ctx_t;

static void* sym(nle_ctx_t *nledl, const char* name) {
  if (nledl->shared) {
    return nleshared_sym(nledl->shared, name);
  } else {
    dlerror(); /* Clear any existing error */
    void* r = dlsym(nledl->dlhandle, name);
    char *error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }
    return r;
  }
}

void
nledl_init(nle_ctx_t *nledl, nle_obs *obs, nle_seeds_init_t *seed_init, int shared)
{ 
  nledl->shared = NULL;
  if (shared) {
#ifdef HASSHARED
    nledl->shared = nleshared_open(nledl->dlpath);
#else
    fprintf(stderr, "Shared mode not supported on this system!\n");
    exit(EXIT_FAILURE);
#endif
  } else {
    nledl->dlhandle = dlopen(nledl->dlpath, RTLD_LAZY);
    if (!nledl->dlhandle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
  }


  nledl->start = sym(nledl, "nle_start");
  nledl->step = sym(nledl, "nle_step");
  nledl->end = sym(nledl, "nle_end");

  nledl->nle_ctx = nledl->start(obs, nledl->ttyrec, seed_init);
}

void
nledl_close(nle_ctx_t *nledl)
{
    nledl->end(nledl->nle_ctx);

    if (nledl->shared) {
      nleshared_close(nledl->shared);
    } else {
      if (dlclose(nledl->dlhandle)) {
          fprintf(stderr, "Error in dlclose: %s\n", dlerror());
          exit(EXIT_FAILURE);
      }

      dlerror();
    }
}

nle_ctx_t *
nle_start(const char *dlpath, nle_obs *obs, FILE *ttyrec,
          nle_seeds_init_t *seed_init, int shared)
{
    /* TODO: Consider getting ttyrec path from caller? */
    struct nledl_ctx *nledl = malloc(sizeof(struct nledl_ctx));
    nledl->ttyrec = ttyrec;
    strncpy(nledl->dlpath, dlpath, sizeof(nledl->dlpath));

    nledl_init(nledl, obs, seed_init, shared);
    return nledl;
};

nle_ctx_t *
nle_step(nle_ctx_t *nledl, nle_obs *obs)
{
    if (!nledl || (!nledl->dlhandle && !nledl->shared) || !nledl->nle_ctx) {
        fprintf(stderr, "Illegal nledl_ctx\n");
        exit(EXIT_FAILURE);
    }

    nledl->step(nledl->nle_ctx, obs);

    return nledl;
}

/* TODO: For a standard reset, we don't need the full close in nle.c.
 * E.g., we could re-use the stack buffer and the nle_ctx_t. */
void
nle_reset(nle_ctx_t *nledl, nle_obs *obs, FILE *ttyrec,
          nle_seeds_init_t *seed_init)
{
    if (nledl->shared) {
      nledl->end(nledl->nle_ctx);
      nleshared_reset(nledl->shared);
      if (ttyrec)
          nledl->ttyrec = ttyrec;
      nledl->nle_ctx = nledl->start(obs, ttyrec, seed_init);
    } else {
      nledl_close(nledl);
      /* Reset file only if not-NULL. */
      if (ttyrec)
          nledl->ttyrec = ttyrec;

      // TODO: Consider refactoring nledl.h such that we expose this init
      // function but drop reset.
      nledl_init(nledl, obs, seed_init, 0);
    }
}

void
nle_end(nle_ctx_t *nledl)
{
    nledl_close(nledl);
    free(nledl);
}

void
nle_set_seed(nle_ctx_t *nledl, unsigned long core, unsigned long disp,
             char reseed)
{
    void (*set_seed)(void *, unsigned long, unsigned long, char);

    set_seed = sym(nledl, "nle_set_seed");

    set_seed(nledl->nle_ctx, core, disp, reseed);
}

void
nle_get_seed(nle_ctx_t *nledl, unsigned long *core, unsigned long *disp,
             char *reseed)
{
    void (*get_seed)(void *, unsigned long *, unsigned long *, char *);

    get_seed = sym(nledl, "nle_get_seed");

    /* Careful here. NetHack has different ideas of what a boolean is
     * than C++ (see global.h and SKIP_BOOLEAN). But one byte should be fine.
     */
    get_seed(nledl->nle_ctx, core, disp, reseed);
}

int
nle_supports_shared(void) {
#ifdef HASSHARED
  return 1;
#else
  return 0;
#endif
}
