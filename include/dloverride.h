/* Copyright (c) Facebook, Inc. and its affiliates. */

/*
 * Mechanism to reset a loaded dynamic library.
 */

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <assert.h>
#include <stdio.h>

#include <dlfcn.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __linux__
#include <elf.h>

#define PAGE_SIZE 4096
#define PAGE_MASK (PAGE_SIZE - 1)
#define PAGE_START(x) ((x) & ~PAGE_MASK)
#define PAGE_OFFSET(x) ((x) &PAGE_MASK)
#define PAGE_END(x) PAGE_START((x) + (PAGE_SIZE - 1))

#if __LP64__
#define Elf_Ehdr Elf64_Ehdr
#define Elf_Phdr Elf64_Phdr

#else /* __LP64__ */
#define Elf_Ehdr Elf32_Ehdr
#define Elf_Phdr Elf32_Phdr
#endif /* __LP64__ */

#elif __APPLE__
#include <mach-o/dyld.h>

#if __LP64__
#define LC_SEGMENT_COMMAND LC_SEGMENT_64
#define MH_MAGIC_NUMBER MH_MAGIC_64

struct macho_header : public mach_header_64 {
};
struct macho_segment_command : public segment_command_64 {
};
struct macho_section : public section_64 {
};
#else /* __LP64__ */
#define LC_SEGMENT_COMMAND LC_SEGMENT
#define MH_MAGIC_NUMBER MH_MAGIC

struct macho_header : public mach_header {
};
struct macho_segment_command : public segment_command {
};
struct macho_section : public section {
};
#endif /* __LP64__ */

#endif /* __linux__, __APPLE__ */

#ifdef __linux__
struct Region {
    uint8_t *data;
    size_t size;
    bool rw;

    uint8_t *
    l() const
    {
        return data;
    }
    uint8_t *
    r() const
    {
        return data + size;
    }

    bool
    intersects(const Region &s) const
    {
        return !(r() <= s.l() || s.r() <= l());
    }

    bool
    operator<(const Region &s) const
    {
        return data < s.data;
    }
};

std::vector<Region>
make_disjoint(const std::vector<Region> &regions)
{
    std::vector<uint8_t *> starts;
    std::vector<uint8_t *> ends;

    std::vector<Region> result;

    for (auto it = regions.rbegin(); it != regions.rend(); ++it) {
        starts.push_back(it->l());
        ends.push_back(it->r());
    }

    std::sort(starts.begin(), starts.end(), std::greater<>());
    std::sort(ends.begin(), ends.end(), std::greater<>());

    int overlap = 1;
    uint8_t *start = starts.back();
    starts.pop_back();
    uint8_t *end;
    bool active;

    while (!ends.empty()) {
        active = overlap > 0;

        if (!starts.empty() && starts.back() <= ends.back()) {
            ++overlap;
            end = starts.back();
            starts.pop_back();
        } else {
            --overlap;
            end = ends.back();
            ends.pop_back();
        }

        if (active && start < end) {
            result.push_back(Region{ start, (size_t) (end - start), false });
        }
        start = end;
    }

    if (!starts.empty()) {
        throw std::runtime_error("Intervals required");
    }

    return result;
}
#endif

class DL
{
  public:
    DL(const char *filename, const char *symbol)
        : handle_(dlopen(filename, RTLD_LAZY))
    {
        if (!handle_) {
            throw std::runtime_error(std::string("dlopen failed on ")
                                     + filename + ": " + dlerror());
        }

        void *ptr = dlsym(handle_, symbol);
        if (!ptr) {
            throw std::runtime_error(dlerror());
        }
        Dl_info dlinfo;
        if (dladdr(ptr, &dlinfo) == 0) {
            throw std::runtime_error("dladdr failed");
        }
        if (!dlinfo.dli_sname) {
            throw std::runtime_error("No matching addr found.");
        }
#ifdef __linux__
        hdr_ = (Elf_Ehdr *) dlinfo.dli_fbase;
        if (memcmp(hdr_->e_ident, ELFMAG, SELFMAG) != 0) {
            throw std::runtime_error("Illegal elf header");
        }

        size_t offset = ~0;

        size_t phoff = hdr_->e_phoff;
        for (size_t i = 0; i != hdr_->e_phnum; ++i) {
            Elf_Phdr *ph = (Elf_Phdr *) ((uint8_t *) hdr_ + phoff);

            if (ph->p_type == PT_LOAD) {
                offset = std::min(offset, (size_t) ph->p_vaddr);
            }
            phs_.push_back(ph);
            phoff += hdr_->e_phentsize;
        }

        baseaddr_ = (uint8_t *) hdr_ - offset;

        std::vector<Region> overlapping_regions;

        for (const Elf_Phdr *ph : phs_) {
            overlapping_regions.push_back(
                Region{ baseaddr_ + ph->p_vaddr, ph->p_memsz, false });
        }

        regions_ = make_disjoint(overlapping_regions);

        for (const Elf_Phdr *ph : phs_) {
            Region region{ baseaddr_ + ph->p_vaddr, ph->p_memsz,
                           ph->p_flags & PF_W ? true : false };

            for (Region &s : regions_) {
                if (region.intersects(s)) {
                    s.rw = region.rw;
                } else if (region.l() < s.r()) {
                    break;
                }
            }
        }

#elif __APPLE__
        hdr_ = (macho_header *) dlinfo.dli_fbase;
        if (hdr_->magic != MH_MAGIC_NUMBER) {
            throw std::runtime_error(
                "Illegal magic integer " + std::to_string(hdr_->magic)
                + ", expected " + std::to_string(MH_MAGIC_NUMBER));
        }
        if (hdr_->filetype != MH_DYLIB) {
            throw std::runtime_error(
                std::string("Expected MH_DYLIB file type but got "
                            + std::to_string(hdr_->filetype)));
        }

        const load_command *cmds = (load_command *) (hdr_ + 1);
        const load_command *cmd = cmds;

        for (uint32_t i = 0; i < hdr_->ncmds; ++i) {
            if (cmd->cmd != LC_SEGMENT_COMMAND)
                continue;

            const auto *seg = (macho_segment_command *) cmd;
            if (seg->nsects)
                segs_.push_back(seg);
            cmd = (const load_command *) (((uint8_t *) cmd) + cmd->cmdsize);
        }

        baseaddr_ = (uint8_t *) hdr_ - segs_[0]->vmaddr;
#endif /* __linux__, __APPLE__ */
    }

    ~DL()
    {
        if (handle_)
            dlclose(handle_);
    }

    DL(DL &&dl) noexcept
    {
        *this = std::move(dl);
    }
    DL(const DL &) = delete;
    DL &operator=(const DL &) = delete;
    DL &
    operator=(DL &&dl) noexcept
    {
        if (this == &dl)
            return *this;
        if (handle_)
            dlclose(handle_);
        handle_ = std::exchange(dl.handle_, nullptr);
#ifdef __linux__
        phs_ = std::move(dl.phs_);
        regions_ = std::move(dl.regions_);
#elif __APPLE__
        segs_ = std::move(dl.segs_);
#endif
        hdr_ = dl.hdr_;
        baseaddr_ = dl.baseaddr_;
        return *this;
    }

#ifdef __linux__
    bool
    is_rw(const Elf_Phdr *ph) const
    {
        return ph->p_type == PT_LOAD && ph->p_flags & PF_R
               && ph->p_flags & PF_W;
    }
    bool
    is_rw(const Region &region) const
    {
        return region.rw;
    }
#elif __APPLE__
    bool
    is_rw(const macho_segment_command *seg) const
    {
        return strcmp(seg->segname, SEG_DATA) == 0;
    }
#endif

    template <typename F>
    void
    for_rw_regions(F &&f)
    {
#ifdef __linux__
        for (const auto &region : regions_) {
#elif __APPLE__
        for (const auto &region : segs_) {
#endif
            if (is_rw(region))
                f(region);
        }
    }

    template <typename T, typename... Ts>
    auto
    func(const char *symbol) -> decltype((T(*)(Ts...)) nullptr)
    {
        void *ptr = dlsym(handle_, symbol);
        if (!ptr) {
            throw std::runtime_error(dlerror());
        }
        return (T(*)(Ts...)) ptr;
    }

#ifdef __linux__
    uint8_t *
    mem_addr(const Elf_Phdr *ph) const
    {
        size_t start = (size_t) (baseaddr_ + ph->p_vaddr);
        return (uint8_t *) start;
    }
    size_t
    mem_size(const Elf_Phdr *ph) const
    {
        return ph->p_memsz;
    }
    uint8_t *
    mem_addr(const Region &region) const
    {
        return region.data;
    }
    size_t
    mem_size(const Region &region) const
    {
        return region.size;
    }
#elif __APPLE__
    uint8_t *
    mem_addr(const macho_segment_command *seg) const
    {
        return baseaddr_ + seg->vmaddr;
    }
    size_t
    mem_size(const macho_segment_command *seg) const
    {
        return seg->vmsize;
    }
#endif

  private:
    void *handle_{ nullptr };
#ifdef __linux__
    const Elf_Ehdr *hdr_;
    std::vector<const Elf_Phdr *> phs_;
    std::vector<Region> regions_;
#elif __APPLE__
    std::vector<const macho_segment_command *> segs_;
    const macho_header *hdr_;
#endif
    uint8_t *baseaddr_;
};
