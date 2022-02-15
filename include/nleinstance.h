/* Copyright (c) Facebook, Inc. and its affiliates. */

#pragma once

/* #define NLE_RESET_DLOPENCLOSE */ /* Enables the old behaviour: reset by
                                       dlclose & dl-re-open. */

#ifndef NLE_RESET_DLOPENCLOSE
#include "dloverride.h"
#else
#include <dlfcn.h>
#endif

extern "C" {
#include "nleobs.h"
}

#ifndef NLE_RESET_DLOPENCLOSE
class Instance
{
  public:
    Instance(const std::string &dlpath, nle_obs *obs, FILE *ttyrec,
             nle_seeds_init_t *seeds_init, nle_settings *settings)
        : dl_(dlpath.c_str(), "nle_start")
    {
        dl_.for_rw_regions([&](const auto &reg) {
            fprintf(stderr, "memcpy out of [%p, %p)\n", dl_.mem_addr(reg),
                    dl_.mem_addr(reg) + dl_.mem_size(reg));

            regions_.emplace_back(dl_.mem_size(reg));
            memcpy(&regions_.back()[0], dl_.mem_addr(reg), dl_.mem_size(reg));
            fprintf(stderr, "1 memcpy done\n");
        });

        start_ = dl_.func<void *, nle_obs *, FILE *, nle_seeds_init_t *,
                          nle_settings *>("nle_start");
        step_ = dl_.func<void *, void *, nle_obs *>("nle_step");
        end_ = dl_.func<void, void *>("nle_end");
        get_seed_ =
            dl_.func<void, void *, unsigned long *, unsigned long *, char *>(
                "nle_get_seed");
        set_seed_ =
            dl_.func<void, void *, unsigned long, unsigned long, char>(
                "nle_set_seed");

        nle_ctx_ = start_(obs, ttyrec, seeds_init, settings);
    }

    ~Instance()
    {
        if (nle_ctx_)
            close();
    }

    void
    step(nle_obs *obs)
    {
        nle_ctx_ = step_(nle_ctx_, obs);
    }

    void
    reset(nle_obs *obs, FILE *ttyrec, nle_seeds_init_t *seeds_init,
          nle_settings *settings)
    {
        end_(nle_ctx_);

        auto it = regions_.begin();
        dl_.for_rw_regions([&](const auto &reg) {
            fprintf(stderr, "memcpy into [%p, %p)\n", dl_.mem_addr(reg),
                    dl_.mem_addr(reg) + dl_.mem_size(reg));

            memcpy(dl_.mem_addr(reg), it->data(), dl_.mem_size(reg));
            ++it;
        });
        nle_ctx_ = start_(obs, ttyrec, seeds_init, settings);
    }

    void
    close()
    {
        end_(nle_ctx_);
        nle_ctx_ = nullptr;
    }

    void
    get_seed(unsigned long *core, unsigned long *disp, char *reseed)
    {
        get_seed_(nle_ctx_, core, disp, reseed);
    }

    void
    set_seed(unsigned long core, unsigned long disp, char reseed)
    {
        set_seed_(nle_ctx_, core, disp, reseed);
    }

  private:
    DL dl_;
    void *nle_ctx_{ nullptr };

    void *(*start_)(nle_obs *, FILE *, nle_seeds_init_t *, nle_settings *);
    void *(*step_)(void *, nle_obs *);
    void (*end_)(void *);
    void (*get_seed_)(void *, unsigned long *, unsigned long *, char *);
    void (*set_seed_)(void *, unsigned long, unsigned long, char);

    std::vector<std::vector<uint8_t> > regions_;
};
#else /* NLE_RESET_DLOPENCLOSE */
class Instance
{
  public:
    Instance(const std::string &dlpath, nle_obs *obs, FILE *ttyrec,
             nle_seeds_init_t *seeds_init, nle_settings *settings)
        : dlpath_(dlpath)
    {
        init();
        nle_ctx_ = start_(obs, ttyrec, seeds_init, settings);
    }

    ~Instance()
    {
        close();
    }

    void
    step(nle_obs *obs)
    {
        nle_ctx_ = step_(nle_ctx_, obs);
    }

    void
    reset(nle_obs *obs, FILE *ttyrec, nle_seeds_init_t *seeds_init,
          nle_settings *settings)
    {
        close();
        init();
        nle_ctx_ = start_(obs, ttyrec, seeds_init, settings);
    }

    void
    close()
    {
        if (nle_ctx_)
            end_(nle_ctx_);
        nle_ctx_ = nullptr;

        if (handle_)
            if (dlclose(handle_))
                throw std::runtime_error(dlerror());
        handle_ = nullptr;
    }

    void
    get_seed(unsigned long *core, unsigned long *disp, char *reseed)
    {
        get_seed_(nle_ctx_, core, disp, reseed);
    }

    void
    set_seed(unsigned long core, unsigned long disp, char reseed)
    {
        set_seed_(nle_ctx_, core, disp, reseed);
    }

  private:
    void
    init()
    {
        void *handle = dlopen(dlpath_.c_str(), RTLD_LAZY | RTLD_NOLOAD);
        if (handle) {
            dlclose(handle);
            throw std::runtime_error(dlpath_ + " is already loaded");
        }
        handle_ = dlopen(dlpath_.c_str(), RTLD_LAZY);
        if (!handle_) {
            throw std::runtime_error(dlerror());
        }

        start_ = (decltype(start_)) get_sym("nle_start");
        step_ = (decltype(step_)) get_sym("nle_step");
        end_ = (decltype(end_)) get_sym("nle_end");
        get_seed_ = (decltype(get_seed_)) get_sym("nle_get_seed");
        set_seed_ = (decltype(set_seed_)) get_sym("nle_set_seed");
    }

    void *
    get_sym(const char *sym)
    {
        dlerror(); /* Clear.*/
        void *result = dlsym(handle_, sym);
        const char *error = dlerror();
        if (error) {
            throw std::runtime_error(error);
        }
        return result;
    }

    const std::string dlpath_;

    void *handle_{ nullptr };
    void *nle_ctx_{ nullptr };

    void *(*start_)(nle_obs *, FILE *, nle_seeds_init_t *, nle_settings *);
    void *(*step_)(void *, nle_obs *);
    void (*end_)(void *);
    void (*get_seed_)(void *, unsigned long *, unsigned long *, char *);
    void (*set_seed_)(void *, unsigned long, unsigned long, char);
};
#endif
