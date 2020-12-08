
#if defined(__linux__) && defined(__x86_64__)

#include <elf.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <string_view>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <unordered_set>

namespace nleshared {

struct Fd {
  int fd = -1;
  Fd() = default;
  Fd(int fd) noexcept : fd(fd) {}
  Fd(const Fd&) = delete;
  Fd(Fd&& n) noexcept {
    fd = std::exchange(n.fd, -1);
  }
  Fd& operator=(const Fd&) = delete;
  Fd& operator=(Fd&& n) noexcept {
    std::swap(fd, n.fd);
    return *this;
  }
  operator int() const noexcept {
    return fd;
  }
  int release() noexcept {
    return std::exchange(fd, -1);
  }
  ~Fd() {
    if (fd != -1) {
      ::close(fd);
    }
  }
  size_t fileSize() const {
    struct ::stat s;
    if (::fstat(fd, &s)) {
      throw std::system_error(errno, std::system_category(), "fstat");
    }
    return s.st_size;
  }
};

struct Mmap {
  std::byte* data_ = nullptr;
  size_t size_ = 0;
  Mmap() = default;
  Mmap(std::byte* data, size_t size) noexcept : data_(data), size_(size) {}
  Mmap(const Mmap&) = delete;
  Mmap(Mmap&& n) noexcept {
    data_ = std::exchange(n.data_, nullptr);
    size_ = n.size_;
  }
  Mmap& operator=(const Mmap&) = delete;
  Mmap& operator=(Mmap&& n) noexcept {
    std::swap(data_, n.data_);
    std::swap(size_, n.size_);
    return *this;
  }
  ~Mmap() {
    if (data_) {
      ::munmap(data_, size_);
    }
  }
  static Mmap make(const Fd& fd, size_t offset, size_t size, bool write, bool shared) {
    errno = 0;
    void* ptr = ::mmap(nullptr, size, write ? PROT_READ | PROT_WRITE : PROT_READ, shared ? MAP_SHARED : MAP_PRIVATE, (int)fd, offset);
    if (!ptr || ptr == MAP_FAILED) {
      throw std::system_error(errno, std::system_category(), "mmap");
    }
    return {(std::byte*)ptr, size};
  }
  static Mmap shared(const Fd& fd, size_t offset, size_t size, bool write) {
    return make(fd, offset, size, write, true);
  }
  static Mmap private_(const Fd& fd, size_t offset, size_t size, bool write) {
    return make(fd, offset, size, write, false);
  }
  operator bool() const noexcept {
    return data_;
  }
  operator std::byte*() const noexcept {
    return data_;
  }
  std::byte* data() noexcept {
    return data_;
  }
  const std::byte* data() const noexcept {
    return data_;
  }
  size_t size() const noexcept {
    return size_;
  }
};

struct RelativeRelocation {
  uint64_t targetOffset;
  uint64_t valueOffset;
};

struct Image {
  Mmap mem;
  std::byte* base = nullptr;

  operator bool() const noexcept {
    return mem;
  }
};

class NHLoader {
  // The raw file data is kept around as ELF headers are generally read from it,
  // and there is no guarantee that the in-memory layout mirrors the file data.
  Mmap fileMem;
  Elf64_Ehdr* hdr;
  // The "base" form of the library, mapped into memory with all symbols resolved.
  Fd sharedFd;
  Image sharedImage;
  // Relocations that must be performed at fork time
  std::vector<RelativeRelocation> forkRelocations;
  // Relocations that must be performed at reset time
  std::vector<RelativeRelocation> resetRelocations;

  size_t initfunc;
  size_t finifunc;
  size_t initarray;
  size_t initarraysz;
  size_t finiarray;
  size_t finiarraysz;

  void load(std::string libPath, const std::unordered_map<std::string, void*>& overrides) {
    Fd diskFd = open(libPath.c_str(), O_RDONLY);

    fileMem = Mmap::shared(diskFd, 0, diskFd.fileSize(), false);

    hdr = (Elf64_Ehdr*)fileMem.data();
    boundsCheck(hdr);

    size_t baseaddr = ~0;
    size_t endaddr = 0;

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD) {
        baseaddr = std::min(baseaddr, (size_t)ph->p_vaddr);
        endaddr = std::max(endaddr, (size_t)(ph->p_vaddr + ph->p_memsz));
      }
    });

    if (baseaddr == (size_t)~0) {
      throw std::runtime_error("No loadable program segments found");
    }

    sharedFd = memfd_create("nethack", 0);
    if ((int)sharedFd < 0) {
      throw std::system_error(errno, std::system_category(), "memfd_create");
    }
    size_t memsize = endaddr - baseaddr;
    if (ftruncate(sharedFd, memsize)) {
      throw std::system_error(errno, std::system_category(), "ftruncate");
    }

    sharedImage.mem = Mmap::shared(sharedFd, 0, memsize, true);
    sharedImage.base = sharedImage.mem.data() - baseaddr;

    // Copy over all initialized data from the file to the base shared image.
    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD) {
        std::memcpy(sharedImage.base + ph->p_vaddr, fileMem.data() + ph->p_offset, ph->p_filesz);
      }
    });
    // And link it.
    link(overrides);
    // We should never write to the shared image again, so make it read-only
    mprotect(sharedImage.mem.data(), sharedImage.mem.size(), PROT_READ);
  }

  void boundsCheck(std::byte* begin, std::byte* end) {
    if (begin < fileMem.data() || end > fileMem.data() + fileMem.size()) {
      throw std::runtime_error("Out of bounds access parsing ELF (corrupt file?)");
    }
  }

  template<typename T>
  void boundsCheck(T* ptr) {
    std::byte* begin = (std::byte*)ptr;
    std::byte* end = begin + sizeof(T);
    boundsCheck(begin, end);
  }

  template<typename F>
  void forPh(F&& f) {
    size_t phoff = hdr->e_phoff;
    for (size_t i = 0; i != hdr->e_phnum; ++i) {
      Elf64_Phdr* ph = (Elf64_Phdr*)(fileMem.data() + phoff);
      boundsCheck(ph);
      f(ph);
      phoff += hdr->e_phentsize;
    }
  }

  // Dynamic linking. Loads required libraries, resolves symbols and performs
  // relocations. Some relocations must also be performed at fork time.
  void link(const std::unordered_map<std::string, void*>& overrides) {
    std::byte* base = sharedImage.base;

    std::byte* symtab = nullptr;
    const char* strtab = nullptr;

    size_t relasz = 0;
    size_t relaent = 0;
    size_t pltrelsz = 0;
    size_t syment = 0;

    forPh([&](Elf64_Phdr* ph) {

      if (ph->p_type == PT_DYNAMIC) {
        Elf64_Dyn* dyn = (Elf64_Dyn*)(fileMem.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(fileMem.data() + ph->p_offset + ph->p_filesz);
        while (dyn < dynEnd) {
          boundsCheck(dyn);

          switch (dyn->d_tag) {
          case DT_SYMTAB:
            symtab = base + dyn->d_un.d_ptr;
            break;
          case DT_STRTAB:
            strtab = (const char*)(base + dyn->d_un.d_ptr);
            break;
          case DT_RELASZ:
            relasz = dyn->d_un.d_val;
            break;
          case DT_RELAENT:
            relaent = dyn->d_un.d_val;
            break;
          case DT_SYMENT:
            syment = dyn->d_un.d_val;
            break;
          case DT_PLTRELSZ:
            pltrelsz = dyn->d_un.d_val;
            break;
          }

          ++dyn;
        }
      }
    });
    forPh([&](Elf64_Phdr* ph) {

      if (ph->p_type == PT_DYNAMIC) {
        Elf64_Dyn* dyn = (Elf64_Dyn*)(fileMem.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(fileMem.data() + ph->p_offset + ph->p_filesz);
        while (dyn < dynEnd) {
          boundsCheck(dyn);

          switch (dyn->d_tag) {
          case DT_NEEDED:
            // TODO: respect RUNPATH
            //       close this dl on dtor?
            dlopen(strtab + dyn->d_un.d_val, RTLD_NOW | RTLD_GLOBAL);
            break;
          }

          ++dyn;
        }
      }
    });

    struct Address {
      bool isRelative;
      uint64_t value;
    };

    std::unordered_map<size_t, std::byte*> symbolAddressMap;

    auto symbolAddress = [&](size_t index) {
      Elf64_Sym* sym = (Elf64_Sym*)(symtab + syment * index);
      if (sym->st_value) {
        return Address{true, sym->st_value};
      }
      auto i = symbolAddressMap.emplace(index, nullptr);
      auto& r = i.first->second;
      if (!i.second) {
        return Address{false, (uint64_t)r};
      }
      std::string name = strtab + sym->st_name;
      //printf("Looking up symbol %s\n", name.c_str());

      auto oi = overrides.find(name);
      if (oi != overrides.end()) {
        r = (std::byte*)oi->second;
      } else {
        r = (std::byte*)dlsym(RTLD_DEFAULT, name.c_str());
        if (!r && ELF64_ST_BIND(sym->st_info) != STB_WEAK) {
          throw std::runtime_error("Symbol " + name + " not found");
        }
      }
      return Address{false, (uint64_t)r};
    };

    auto copy64 = [&](std::byte* dst, Address addr) {
      if (addr.isRelative) {
        forkRelocations.push_back({uint64_t(dst - base), addr.value});
      }
      std::byte* value = addr.isRelative ? base + addr.value : (std::byte*)addr.value;
      std::memcpy(dst, &value, sizeof(value));
    };

    auto doRela = [&](Elf64_Rela* rela) {
      auto type = ELF64_R_TYPE(rela->r_info);
      auto sym = ELF64_R_SYM(rela->r_info);
      std::byte* address = base + rela->r_offset;
      auto addend = rela->r_addend;
      switch (type) {
      case R_X86_64_JUMP_SLOT:
      case R_X86_64_GLOB_DAT:
        copy64(address, symbolAddress(sym));
        break;
      case R_X86_64_RELATIVE:
        copy64(address, Address{true, (uint64_t)addend});
        break;
      case R_X86_64_64: {
        auto a = symbolAddress(sym);
        a.value += addend;
        copy64(address, a);
        break;
      }
      default:
        throw std::runtime_error("Unsupported relocation type " + std::to_string(type));
      }
    };

    initfunc = 0;
    finifunc = 0;
    initarray = 0;
    initarraysz = 0;
    finiarray = 0;
    finiarraysz = 0;

    forPh([&](Elf64_Phdr* ph) {

      if (ph->p_type == PT_DYNAMIC) {
        Elf64_Dyn* dyn = (Elf64_Dyn*)(fileMem.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(fileMem.data() + ph->p_offset + ph->p_filesz);
        while (dyn < dynEnd) {
          boundsCheck(dyn);

          switch (dyn->d_tag) {
          case DT_RELA: {
            if (relasz > 0 && relaent > 0) {
              for (size_t i = 0; i != relasz / relaent; ++i) {
                Elf64_Rela* rela = (Elf64_Rela*)(base + dyn->d_un.d_ptr + relaent * i);
                doRela(rela);
              }
            }
            break;
          }
          case DT_REL:
            throw std::runtime_error("RELA not implemented");
            break;
          case DT_PLTREL: {
            if (dyn->d_un.d_val != DT_RELA) {
              throw std::runtime_error("Unsupported value for DT_PLTREL");
            }
            break;
          }
          case DT_JMPREL: {
            if (relasz > 0 && relaent > 0) {
              for (size_t i = 0; i != pltrelsz / relaent; ++i) {
                Elf64_Rela* rela = (Elf64_Rela*)(base + dyn->d_un.d_ptr + relaent * i);
                doRela(rela);
              }
            }
            break;
          }
          case DT_INIT:
            initfunc = dyn->d_un.d_ptr;
            break;
          case DT_FINI:
            finifunc = dyn->d_un.d_ptr;
            break;
          case DT_INIT_ARRAY:
            initarray = dyn->d_un.d_ptr;
            break;
          case DT_INIT_ARRAYSZ:
            initarraysz = dyn->d_un.d_val;
            break;
          case DT_FINI_ARRAY:
            finiarray = dyn->d_un.d_ptr;
            break;
          case DT_FINI_ARRAYSZ:
            finiarraysz = dyn->d_un.d_val;
            break;
          }

          ++dyn;
        }
      }
    });

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        uint64_t begin = ph->p_vaddr;
        uint64_t end = ph->p_vaddr + ph->p_memsz;
        for (auto [dst, offset] : forkRelocations) {
          if (dst >= begin && dst < end) {
            resetRelocations.push_back({dst, offset});
          }
        }
      }
    });

  }

public:

  NHLoader(std::string libPath, const std::unordered_map<std::string, void*>& overrides) {
    load(libPath, overrides);
  }

  // Convenience class for referencing symbols.
  template<typename T>
  struct Symbol {
    size_t offset;

    template<typename Instance>
    auto* resolve(Instance& i) {
      return (T*)(void*)(i.image.base + offset);
    }
  };
  template<typename R, typename... Args>
  struct Symbol<R(Args...)> {
    size_t offset;

    template<typename Instance>
    R operator()(Instance& i, Args... args) {
      return i.call(*this, args...);
    }

    template<typename Instance>
    auto* resolve(Instance& i) {
      return (R(*)(Args...))(void*)(i.image.base + offset);
    }
  };

  // One instance of the shared library, with its own address and data.
  struct Instance {
    NHLoader* loader = nullptr;
    Image image;
    Instance() = default;
    Instance(NHLoader& loader) : loader(&loader) {}
    Instance(Instance&&) = default;
    Instance& operator=(Instance&&) = default;
    // Resets the writable parts of the image to their initial values.
    // Does not call finalization functions or clean up any other resources
    void reset() {
      loader->reset(*this);
    }
    // Call global initialization functions (eg global constructors)
    void init() {
      if (loader->initfunc) {
        ((void(*)())(image.base + loader->initfunc))();
      }
      if (loader->initarray) {
        for (size_t i = 0; i != loader->initarraysz / sizeof(void*); ++i) {
          ((void(*)())((void**)(image.base + loader->initarray))[i])();
        }
      }
    }
    // Call global finalization functions (eg global destructors)
    void fini() {
      if (loader->finiarray) {
        for (size_t i = 0; i != loader->finiarraysz / sizeof(void*); ++i) {
          ((void(*)())((void**)(image.base + loader->finiarray))[i])();
        }
      }
      if (loader->finifunc) {
        ((void(*)())(image.base + loader->finifunc))();
      }
    }
    template<typename Sig, typename... Args>
    auto call(Symbol<Sig> func, Args&&... args) {
      return ((Sig*)(void*)(image.base + func.offset))(std::forward<Args>(args)...);
    }
  };

  template<typename Sig>
  Symbol<Sig> symbol(std::string_view name) {
    uint64_t offset = hdr->e_shoff;
    size_t n = hdr->e_shnum;
    offset = hdr->e_shoff;
    for (size_t i = 0; i != n; ++i) {
      Elf64_Shdr* shdr = (Elf64_Shdr*)(fileMem.data() + offset);
      boundsCheck(shdr);

      if (shdr->sh_type == SHT_SYMTAB) {

        size_t strtab = ((Elf64_Shdr*)(fileMem.data() + hdr->e_shoff + hdr->e_shentsize * shdr->sh_link))->sh_offset;

        auto begin = shdr->sh_offset;
        auto end = begin + shdr->sh_size;
        for (auto i = begin; i < end; i += shdr->sh_entsize) {
          Elf64_Sym* sym = (Elf64_Sym*)(fileMem.data() + i);
          boundsCheck(sym);

          std::string_view sname = (const char*)(fileMem.data() + strtab + sym->st_name);
          if (sname == name) {
            return {(size_t)sym->st_value};
          }
        }
      }

      offset += hdr->e_shentsize;
    }

    throw std::runtime_error("Could not find symbol " + std::string(name));
  }

  // Create a new instance of the shared library. Most memory will initially
  // be memory mapped from the shared image with copy-on-write semantics.
  Instance fork() {
    Mmap mem = Mmap::private_(sharedFd, 0, sharedImage.mem.size(), true);

    std::byte* base = (std::byte*)mem.data() + (sharedImage.base - sharedImage.mem.data());

    for (auto& v : forkRelocations) {
      std::byte* dst = base + v.targetOffset;
      std::byte* value = base + v.valueOffset;
      std::memcpy(dst, &value, sizeof(value));
    }

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD) {
        int flags = 0;
        if (ph->p_flags & PF_R) {
          flags |= PROT_READ;
        }
        if (ph->p_flags & PF_W) {
          flags |= PROT_WRITE;
        }
        if (ph->p_flags & PF_X) {
          flags |= PROT_EXEC;
        }
        mprotect(base + ph->p_vaddr, ph->p_memsz, flags);
      }
    });

    Instance i(*this);
    i.image.mem = std::move(mem);
    i.image.base = base;
    return i;
  }

  void reset(Instance& i) {
    // Reset memory in writable segments to initial values
    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        std::memcpy(i.image.base + ph->p_vaddr, sharedImage.base + ph->p_vaddr, ph->p_memsz);
      }
    });
    // Perform any necessary relocations in the data we just copied to fit
    // our base address.
    for (auto& v : resetRelocations) {
      std::byte* dst = i.image.base + v.targetOffset;
      std::byte* value = i.image.base + v.valueOffset;
      std::memcpy(dst, &value, sizeof(value));
    }
  }

};

namespace env {

void init(struct Env*);

struct Env {
  NHLoader::Instance instance;
  Env(NHLoader::Instance instance) : instance(std::move(instance)) {
    this->instance.init();
    init(this);
  }
  ~Env() {
    instance.fini();
  }
  void reset() {
    instance.fini();
    instance.reset();
    instance.init();
  }

  std::string cwd;
  NHLoader::Symbol<void(int)> nethack_exit;
};

thread_local Env* current = nullptr;

void init(Env* env) {
  current = env;

  env->nethack_exit = env->instance.loader->symbol<void(int)>("nethack_exit");
}

void nethack_exit(int status) {
  current->nethack_exit(current->instance, status);
  std::abort();
}

int has_colors() {
  return 1;
}

void exit(int status) {
  nethack_exit(status);
}
void abort() {
  nethack_exit(-1);
}
void popen() {
  nethack_exit(-1);
}
void fork() {
  nethack_exit(-1);
}
void sleep() {
  nethack_exit(-1);
}
void execl() {
  nethack_exit(-1);
}

int chdir(const char* path) {
  current->cwd = path;
  return 0;
}
int rename(const char*, const char*) {
  errno = ENOENT;
  return -1;
}
int unlink(const char*) {
  errno = ENOENT;
  return -1;
}

// List of files that we allow read-only access to
const std::unordered_set<std::string_view> allowedFiles = {
  "/dev/urandom"
};

// List of files in nethackdir that we allow read-only access to
const std::unordered_set<std::string_view> nethackdirFiles = {
  "sysconf", "nhdat"
};

int creat(const char* path, int) {
  return memfd_create(path, 0);
}
int open(const char* path, int flags, ...) {
  if (allowedFiles.find(path) != allowedFiles.end()) {
    return open(path, O_RDONLY);
  }
  if (nethackdirFiles.find(path) != nethackdirFiles.end()) {
    return ::open((current->cwd + "/" + path).c_str(), O_RDONLY);
  }
  if (flags & O_CREAT) {
    return memfd_create(path, 0);
  } else {
    errno = ENOENT;
    return -1;
  }
}
FILE* fopen(const char* path, const char* mode) {
  if (allowedFiles.find(path) != allowedFiles.end()) {
    return ::fopen(path, "rb");
  }
  if (nethackdirFiles.find(path) != nethackdirFiles.end()) {
    return ::fopen((current->cwd + "/" + path).c_str(), "rb");
  }
  for (const char* c = mode; *c; ++c) {
    if (*c == 'w' || *c == 'a') {
      return ::fdopen(memfd_create(path, 0), mode);
    }
  }
  errno = ENOENT;
  return nullptr;
}
int setuid(uid_t) {
  errno = EPERM;
  return -1;
}


std::unordered_map<std::string, void*> makeOverrides() {
  std::unordered_map<std::string, void*> overrides;

  auto ovr = [&](std::string name, auto* f) {
    overrides[name] = (void*)f;
  };

  ovr("has_colors", env::has_colors); // Why is this necessary?

  // Functions that are not allowed and will end the game
  ovr("exit", env::exit);
  ovr("_exit", env::exit);
  ovr("abort", env::abort);
  ovr("popen", env::popen);
  ovr("fork", env::fork);
  ovr("sleep", env::sleep);
  ovr("execl", env::execl);

  // Functions that we replace
  ovr("chdir", env::chdir);
  ovr("rename", env::rename);
  ovr("creat", env::creat);
  ovr("open", env::open);
  ovr("rename", env::rename);
  ovr("unlink", env::unlink);
  ovr("fopen", env::fopen);
  ovr("setuid", env::setuid);
//    ovr("getpwnam", rep::getpwnam);
//    ovr("getpwuid", rep::getpwuid);

  return overrides;
}

std::mutex loaderMutex;
std::unordered_map<std::string, std::optional<NHLoader>> loaderMap;
NHLoader& getLoader(const char* dlpath) {
  std::lock_guard l(loaderMutex);
  auto& loader = loaderMap[dlpath];
  if (!loader) {
    loader.emplace(dlpath, makeOverrides());
  }
  return *loader;
}

}

extern "C" void* nleshared_open(const char *dlpath) {
  return new env::Env{env::getLoader(dlpath).fork()};
}
extern "C" void nleshared_close(void* handle) {
  env::Env* env = (env::Env*)handle;
  delete env;
}
extern "C" void nleshared_reset(void* handle) {
  env::Env* env = (env::Env*)handle;
  env->reset();
}
extern "C" void* nleshared_sym(void* handle, const char* symname) {
  env::Env* env = (env::Env*)handle;
  return (void*)env->instance.loader->symbol<void()>(symname).resolve(env->instance);
}
extern "C" void nleshared_set_current(void* handle) {
  env::current = (env::Env*)handle;
}
extern "C" int nleshared_supported() {
  return true;
}

}

#else

#include <cstdlib>
#include <stdexcept>

extern "C" void* nleshared_open(const char *dlpath) {
  throw std::runtime_error("NLE shared not supported on this platform");
}
extern "C" void nleshared_close(void* handle) {
  std::abort();
}
extern "C" void nleshared_reset(void* handle) {
  std::abort();
}
extern "C" void* nleshared_sym(void* handle, const char* symname) {
  std::abort();
}
extern "C" void nleshared_set_current(void*) {
  std::abort();
}
extern "C" int nleshared_supported() {
  return false;
}
#endif
