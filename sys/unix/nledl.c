
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nledl.h"

void
nledl_init(nledl_ctx *nledl, nle_obs *obs, nle_seeds_init_t *seed_init,
           nle_settings *settings)
{
    void *handle = dlopen(nledl->dlpath, RTLD_LAZY | RTLD_NOLOAD);
    if (handle) {
        dlclose(handle);
        fprintf(stderr,
                "failure in nledl_init: library %s is already loaded\n",
                nledl->dlpath);
        exit(EXIT_FAILURE);
    }

    nledl->dlhandle = dlopen(nledl->dlpath, RTLD_LAZY);

    if (!nledl->dlhandle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    dlerror(); /* Clear any existing error */

    void *(*start)(nle_obs *, FILE *, nle_seeds_init_t *, nle_settings *);
    start = dlsym(nledl->dlhandle, "nle_start");
    nledl->nle_ctx = start(obs, nledl->ttyrec, seed_init, settings);

    char *error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

    nledl->step = dlsym(nledl->dlhandle, "nle_step");

    error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }
}

void
nledl_close(nledl_ctx *nledl)
{
    void (*end)(void *);

    end = dlsym(nledl->dlhandle, "nle_end");
    end(nledl->nle_ctx);

    if (dlclose(nledl->dlhandle)) {
        fprintf(stderr, "Error in dlclose: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    dlerror();
}

nledl_ctx *
nle_start(const char *dlpath, nle_obs *obs, FILE *ttyrec,
          nle_seeds_init_t *seed_init, nle_settings *settings)
{
    /* TODO: Consider getting ttyrec path from caller? */
    struct nledl_ctx *nledl = malloc(sizeof(struct nledl_ctx));
    nledl->ttyrec = ttyrec;
    strncpy(nledl->dlpath, dlpath, sizeof(nledl->dlpath));

    nledl_init(nledl, obs, seed_init, settings);
    return nledl;
};

nledl_ctx *
nle_step(nledl_ctx *nledl, nle_obs *obs)
{
    if (!nledl || !nledl->dlhandle || !nledl->nle_ctx) {
        fprintf(stderr, "Illegal nledl_ctx\n");
        exit(EXIT_FAILURE);
    }

    nledl->step(nledl->nle_ctx, obs);

    return nledl;
}

/* TODO: For a standard reset, we don't need the full close in nle.c.
 * E.g., we could re-use the stack buffer and the nledl_ctx. */
void
nle_reset(nledl_ctx *nledl, nle_obs *obs, FILE *ttyrec,
          nle_seeds_init_t *seed_init, nle_settings *settings)
{
    nledl_close(nledl);
    /* Reset file only if not-NULL. */
    if (ttyrec)
        nledl->ttyrec = ttyrec;

    // TODO: Consider refactoring nledl.h such that we expose this init
    // function but drop reset.
    nledl_init(nledl, obs, seed_init, settings);
}

void
nle_end(nledl_ctx *nledl)
{
    nledl_close(nledl);
    free(nledl);
}

#ifdef NLE_ALLOW_SEEDING
void
nle_set_seed(nledl_ctx *nledl, unsigned long core, unsigned long disp,
             char reseed)
{
    void (*set_seed)(void *, unsigned long, unsigned long, char);

    set_seed = dlsym(nledl->dlhandle, "nle_set_seed");

    char *error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

    set_seed(nledl->nle_ctx, core, disp, reseed);
}

void
nle_get_seed(nledl_ctx *nledl, unsigned long *core, unsigned long *disp,
             char *reseed)
{
    void (*get_seed)(void *, unsigned long *, unsigned long *, char *);

    get_seed = dlsym(nledl->dlhandle, "nle_get_seed");

    char *error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

    /* Careful here. NetHack has different ideas of what a boolean is
     * than C++ (see global.h and SKIP_BOOLEAN). But one byte should be fine.
     */
    get_seed(nledl->nle_ctx, core, disp, reseed);
}
#endif


int 
nle_save(nledl_ctx *nledl)
{   
    int success;
    void *(*dosave0)();

    dosave0 = dlsym(nledl->dlhandle, "dosave0");
    success = dosave0();

    return success;
}