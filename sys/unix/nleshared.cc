
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

struct NHLoader {
  std::vector<std::byte> data;
  Elf64_Ehdr* hdr;
  size_t baseaddr;
  size_t endaddr;
  size_t size;
  int fd;
  void* ptr;
  std::byte* sharedbase;
  std::vector<std::pair<uint64_t, uint64_t>> relocations;
  std::vector<std::pair<uint64_t, uint64_t>> writableRelocations;
  size_t initfunc;
  size_t finifunc;
  size_t initarray;
  size_t initarraysz;
  size_t finiarray;
  size_t finiarraysz;
  NHLoader(std::string libPath, const std::unordered_map<std::string, void*>& overrides) {
    FILE* f = fopen(libPath.c_str(), "rb");
    if (!f) {
      throw std::runtime_error("Failed to open '" + libPath + "' for reading");
    }
    fseek(f, 0, SEEK_END);
    data.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    fread(data.data(), data.size(), 1, f);
    fclose(f);

    hdr = (Elf64_Ehdr*)data.data();
    boundsCheck(hdr);

    baseaddr = ~0;
    endaddr = 0;
    size = 0;

    forPh([&](Elf64_Phdr* ph) {;

      printf("ph type %#x at [%p, %p) (fs %#x)\n", ph->p_type, (void*)ph->p_vaddr, (void*)(ph->p_vaddr + ph->p_memsz), ph->p_filesz);

      if (ph->p_type == PT_LOAD) {
        baseaddr = std::min(baseaddr, (size_t)ph->p_vaddr);
        endaddr = std::max(endaddr, (size_t)(ph->p_vaddr + ph->p_memsz));
      }

    });

    if (baseaddr == (size_t)~0) {
      throw std::runtime_error("No loadable program segments found");
    }

    size = endaddr - baseaddr;

    printf("memsize is %ldM\n", size / 1024 / 1024);

    fd = memfd_create("nethack", 0);
    if (fd < 0) {
      throw std::system_error(errno, std::system_category(), "memfd_create");
    }
    if (ftruncate(fd, size)) {
      throw std::system_error(errno, std::system_category(), "ftruncate");
    }

    ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (!ptr || ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to allocate memory for binary image");
    }

    std::byte* base = (std::byte*)ptr - baseaddr;
    sharedbase = base;

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD) {
        std::memcpy(base + ph->p_vaddr, data.data() + ph->p_offset, ph->p_filesz);
      }
    });
    link(overrides);
    mprotect(ptr, size, PROT_READ);
  }

  void boundsCheck(std::byte* begin, std::byte* end) {
    if (begin < data.data() || end > data.data() + data.size()) {
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
    auto phoff = hdr->e_phoff;
    for (size_t i = 0; i != hdr->e_phnum; ++i) {
      Elf64_Phdr* ph = (Elf64_Phdr*)(data.data() + phoff);
      boundsCheck(ph);

      f(ph);

      phoff += hdr->e_phentsize;
    }
  }

  template<typename T>
  struct Function;
  template<typename R, typename... Args>
  struct Function<R(Args...)> {
    size_t offset;

    template<typename Instance>
    R operator()(Instance& i, Args... args) {
      return i.call(*this, args...);
    }

    template<typename Instance>
    auto* resolve(Instance& i) {
      return (R(*)(Args...))(void*)(i.base + offset);
    }
  };

  struct Instance {
    NHLoader& loader;
    void* ptr;
    std::byte* base;
    Instance(NHLoader& loader) : loader(loader) {}
    Instance(const Instance&) = delete;
    Instance(Instance&& n) : loader(n.loader) {
      ptr = std::exchange(n.ptr, nullptr);
      base = std::exchange(n.base, nullptr);
    }
    ~Instance() {
      if (ptr) {
        loader.free(*this);
      }
    }
    void reset() {
      loader.reset(*this);
    }
    void init() {
      if (loader.initfunc) {
        ((void(*)())(base + loader.initfunc))();
      }
      if (loader.initarray) {
        for (size_t i = 0; i != loader.initarraysz / sizeof(void*); ++i) {
          ((void(*)())((void**)(base + loader.initarray))[i])();
        }
      }
    }
    void fini() {
      if (loader.finiarray) {
        for (size_t i = 0; i != loader.finiarraysz / sizeof(void*); ++i) {
          ((void(*)())((void**)(base + loader.finiarray))[i])();
        }
      }
      if (loader.finifunc) {
        ((void(*)())(base + loader.finifunc))();
      }
    }
    template<typename Sig, typename... Args>
    auto call(Function<Sig> func, Args&&... args) {
      return ((Sig*)(void*)(base + func.offset))(std::forward<Args>(args)...);
    }
  };

  void link(const std::unordered_map<std::string, void*>& overrides) {

    std::byte* base = sharedbase;

    printf("Loaded at base %p\n", base);

    std::byte* symtab = nullptr;
    const char* strtab = nullptr;

    size_t relasz = 0;
    size_t relaent = 0;
    size_t pltrelsz = 0;
    size_t syment = 0;

    forPh([&](Elf64_Phdr* ph) {

      if (ph->p_type == PT_DYNAMIC) {
        Elf64_Dyn* dyn = (Elf64_Dyn*)(data.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(data.data() + ph->p_offset + ph->p_filesz);
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
        Elf64_Dyn* dyn = (Elf64_Dyn*)(data.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(data.data() + ph->p_offset + ph->p_filesz);
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
      printf("Looking up symbol %s\n", name.c_str());

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
        relocations.emplace_back(dst - base, addr.value);
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
        Elf64_Dyn* dyn = (Elf64_Dyn*)(data.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(data.data() + ph->p_offset + ph->p_filesz);
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
            if (dyn->d_un.d_val == DT_RELA) {
              //printf("plt rela\n");
            } else {
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
        for (auto [dst, offset] : relocations) {
          if (dst >= begin && dst < end) {
            writableRelocations.emplace_back(dst, offset);
          }
        }
      }
    });

  }

  template<typename Sig>
  Function<Sig> symbol(std::string name) {
    uint64_t offset = hdr->e_shoff;
    size_t n = hdr->e_shnum;
    offset = hdr->e_shoff;
    for (size_t i = 0; i != n; ++i) {
      Elf64_Shdr* shdr = (Elf64_Shdr*)(data.data() + offset);
      boundsCheck(shdr);

      if (shdr->sh_type == SHT_SYMTAB) {

        size_t strtab = ((Elf64_Shdr*)(data.data() + hdr->e_shoff + hdr->e_shentsize * shdr->sh_link))->sh_offset;

        auto begin = shdr->sh_offset;
        auto end = begin + shdr->sh_size;
        for (auto i = begin; i < end; i += shdr->sh_entsize) {
          Elf64_Sym* sym = (Elf64_Sym*)(data.data() + i);
          boundsCheck(sym);

          //printf("sym->st_name is %d\n", sym->st_name);

          std::string_view sname = (const char*)(data.data() + strtab + sym->st_name);
          if (sname == name) {
            return {(size_t)sym->st_value};
            //printf("found symbol %s\n", std::string(sname).c_str());
          }
        }
      }

      offset += hdr->e_shentsize;
    }

    throw std::runtime_error("Could not find symbol " + name);
  }

  Instance fork() {

    void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (!ptr || ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to allocate memory for binary image");
    }

    std::byte* base = (std::byte*)ptr - baseaddr;

    std::unordered_map<size_t, bool> pagesTouched;

    printf("There are %d relocations\n", relocations.size());

    for (auto& v : relocations) {
      std::byte* dst = base + v.first;
      std::byte* value = base + v.second;
      pagesTouched[(size_t)dst / 0x1000] = true;

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

    printf("Touched %d/%d pages\n", pagesTouched.size(), size / 0x1000);

    int nWritableTouched = 0;
    int nWritableTotal = 0;

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        for (std::byte* p = base + ph->p_vaddr; p < base + ph->p_vaddr + ph->p_memsz; p += 0x1000) {
          if (pagesTouched[(size_t)p / 0x1000]) {
            ++nWritableTouched;
          }
          ++nWritableTotal;
        }
      }
    });

    printf("Writable touched %d/%d pages\n", nWritableTouched, nWritableTotal);

    printf("success!\n");

    printf("forked to base %p\n", base);

    Instance i(*this);
    i.ptr = ptr;
    i.base = base;
    return i;
  }

  void reset(Instance& i) {
    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        std::memcpy(i.base + ph->p_vaddr, sharedbase + ph->p_vaddr, ph->p_memsz);
      }
    });
    for (auto& v : writableRelocations) {
      std::byte* dst = i.base + v.first;
      std::byte* value = i.base + v.second;
      std::memcpy(dst, &value, sizeof(value));
    }
  }

  void free(Instance& i) {
    ::munmap(i.ptr, size);
  }

};

namespace rep {

struct Env {
  NHLoader::Instance instance;
  Env(NHLoader::Instance instance) : instance(std::move(instance)) {
    this->instance.init();
  }
  ~Env() {
    instance.fini();
  }
  void reset() {
    instance.fini();
    instance.reset();
    instance.init();
  }
};

int has_colors() {
  return 1;
}


void exit(int exitcode) {
  printf("exit %d\n", exitcode);
  throw std::runtime_error("Exit!?");
}

void abort() {
  printf("abort called");
  throw std::runtime_error("abort called");
}

void popen() {
  throw std::runtime_error("popen called");
}
void fork() {
  throw std::runtime_error("fork called");
}
void sleep() {
  throw std::runtime_error("sleep called");
}
void execl() {
  throw std::runtime_error("execl called");
}

std::string nethackdir;
std::once_flag nethackdirflag;

int chdir(const char* path) {
  //printf("chdir %s\n", path);
  std::call_once(nethackdirflag, [&]() {
    nethackdir = path;
  });
  return 0;
}
int rename(const char* src, const char* dst) {
  errno = ENOENT;
  return -1;
}
int unlink(const char* path) {
  errno = ENOENT;
  return -1;
}
int creat(const char* path, int) {
  //printf("creat %s\n", path);
  return memfd_create(path, 0);
}

int open(const char* path, int flags, ...) {
  //printf("open %s\n", path);
  if (!strcmp(path, "sysconf") || !strcmp(path, "nhdat")) {
    //printf("nethackdir is %s\n", nethackdir.c_str());
    return ::open((nethackdir + "/" + path).c_str(), O_RDONLY);
  }
  if (flags & O_CREAT) {
    return memfd_create(path, 0);
  } else {
    errno = ENOENT;
    return -1;
  }
}
FILE* fopen(const char* path, const char* mode) {
  //printf("fopen %s\n", path);
  if (!strcmp(path, "/dev/urandom")) {
    return ::fopen(path, mode);
  }
  if (!strcmp(path, "sysconf") || !strcmp(path, "nhdat")) {
    //printf("nethackdir is %s\n", nethackdir.c_str());
    return ::fopen((nethackdir + "/" + path).c_str(), "rb");
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

}

struct NLEShared {
  std::string dlpath;
  NHLoader loader;

  NLEShared(std::string dlpath) : dlpath(dlpath), loader(dlpath, makeOverrides()) {}

  std::unordered_map<std::string, void*> makeOverrides() {
    std::unordered_map<std::string, void*> overrides;

    auto ovr = [&](std::string name, auto* f) {
      overrides[name] = (void*)f;
    };

    ovr("has_colors", rep::has_colors); // Why is this necessary?

    // Functions that are not allowed and will end the game
    ovr("exit", rep::exit);
    ovr("_exit", rep::exit);
    ovr("abort", rep::abort);
    ovr("popen", rep::popen);
    ovr("fork", rep::fork);
    ovr("sleep", rep::sleep);
    ovr("execl", rep::execl);

    // Functions that we replace
    ovr("chdir", rep::chdir);
    ovr("rename", rep::rename);
    ovr("creat", rep::creat);
    ovr("open", rep::open);
    ovr("rename", rep::rename);
    ovr("unlink", rep::unlink);
    ovr("fopen", rep::fopen);
    ovr("setuid", rep::setuid);
//    ovr("getpwnam", rep::getpwnam);
//    ovr("getpwuid", rep::getpwuid);

    return overrides;
  }

};

extern "C" void* nleshared_open(const char *dlpath) {
  static NLEShared shared(dlpath);
  if (shared.dlpath != dlpath) {
    throw std::runtime_error("nleshared only supports one dlpath at the moment");
  }
  return new rep::Env{shared.loader.fork()};
}
extern "C" void nleshared_close(void* handle) {
  rep::Env* env = (rep::Env*)handle;
  delete env;
}
extern "C" void nleshared_reset(void* handle) {
  rep::Env* env = (rep::Env*)handle;
  env->reset();
}
extern "C" void* nleshared_sym(void* handle, const char* symname) {
  rep::Env* env = (rep::Env*)handle;
  return (void*)env->instance.loader.symbol<void()>(symname).resolve(env->instance);
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
#endif
