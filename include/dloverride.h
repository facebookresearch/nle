/* Copyright (c) Facebook, Inc. and its affiliates. */

/*
 * Mechanism to reset a loaded dynamic library.
 */

#pragma once

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
            segs_.push_back(ph);

            phoff += hdr_->e_phentsize;
        }

        baseaddr_ = (uint8_t *) hdr_ - offset;

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
        segs_ = std::move(dl.segs_);
        hdr_ = dl.hdr_;
        baseaddr_ = dl.baseaddr_;
        return *this;
    }

#ifdef __linux__
    bool
    is_overridable(const Elf_Phdr *ph) const
    {
        return ph->p_type == PT_LOAD && ph->p_flags & PF_R
               && ph->p_flags & PF_W;
    }
#elif __APPLE__
    bool
    is_overridable(const macho_segment_command *seg) const
    {
        return strcmp(seg->segname, SEG_DATA) == 0;
    }
#endif

    template <typename F>
    void
    for_changing_sections(F &&f)
    {
        for (const auto *seg : segs_) {
            if (is_overridable(seg))
                f(seg);
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
        size_t start = PAGE_END((size_t) (baseaddr_ + ph->p_vaddr));
        return (uint8_t *) start;
    }
    size_t
    mem_size(const Elf_Phdr *ph) const
    {
        return ph->p_memsz;
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
    std::vector<const Elf_Phdr *> segs_;
    const Elf_Ehdr *hdr_;
#elif __APPLE__
    std::vector<const macho_segment_command *> segs_;
    const macho_header *hdr_;
#endif
    uint8_t *baseaddr_;
};
