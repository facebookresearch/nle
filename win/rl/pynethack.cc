/* Copyright (c) Facebook, Inc. and its affiliates. */
#include <atomic>
#include <cstdio>
#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// "digit" is declared in both Python's longintrepr.h and NetHack's extern.h.
#define digit nethack_digit

extern "C" {
#include "hack.h"
#include "permonst.h"
#include "pm.h" // File generated during NetHack compilation.
#include "rm.h"
}

extern "C" {
#include "nledl.h"
}

// Undef name clashes between NetHack and Python.
#undef yn
#undef min
#undef max

namespace py = pybind11;
using namespace py::literals;

template <typename T>
T *
checked_conversion(py::handle h, const std::vector<ssize_t> &shape)
{
    if (h.is_none())
        return nullptr;
    py::array array = py::array::ensure(h);
    if (!array)
        throw std::runtime_error("Numpy array required");

    // We don't use py::array_t<T> (or <T, 0>) above as that still
    // causes conversions to "larger" types.

    // TODO: Better error messages here and below.
    if (!array.dtype().is(py::dtype::of<T>()))
        throw std::runtime_error("Numpy array of right type required");

    py::buffer_info buf = array.request();

    if (buf.ndim != shape.size())
        throw std::runtime_error("array has wrong number of dims");
    if (!std::equal(shape.begin(), shape.end(), buf.shape.begin()))
        throw std::runtime_error("Array has wrong shape");
    if (!(array.flags() & py::array::c_style))
        throw std::runtime_error("Array isn't C contiguous");

    return static_cast<T *>(buf.ptr);
}

class Nethack
{
  public:
    Nethack(std::string dlpath, std::string ttyrec)
        : dlpath_(std::move(dlpath)), obs_{},
          ttyrec_(std::fopen(ttyrec.c_str(), "a"), std::fclose)
    {
        if (!ttyrec_) {
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, ttyrec.c_str());
            throw py::error_already_set();
        }
    }
    ~Nethack()
    {
        close();
    }
    void
    step(int action)
    {
        if (!nle_)
            throw std::runtime_error("step called without reset()");
        if (obs_.done)
            throw std::runtime_error("Called step on finished NetHack");
        obs_.action = action;
        nle_ = nle_step(nle_, &obs_);
    }
    bool
    done()
    {
        return obs_.done;
    }

    void
    reset()
    {
        reset(nullptr);
    }

    void
    reset(std::string ttyrec)
    {
        FILE *f = std::fopen(ttyrec.c_str(), "a");
        if (!f) {
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, ttyrec.c_str());
            throw py::error_already_set();
        }
        // Reset environment, then close original FILE. Cannot use freopen
        // as the game may still need to write to the original but reset()
        // wants to get the new one already.
        reset(f);
        ttyrec_.reset(f);
    }

    void
    set_buffers(py::object glyphs, py::object chars, py::object colors,
                py::object specials, py::object blstats, py::object message,
                py::object program_state, py::object internal,
                py::object inv_glyphs, py::object inv_letters,
                py::object inv_oclasses, py::object inv_strs,
                py::object screen_descriptions, py::object tty_chars,
                py::object tty_colors, py::object tty_cursor, py::object misc)
    {
        std::vector<ssize_t> dungeon{ ROWNO, COLNO - 1 };
        obs_.glyphs = checked_conversion<int16_t>(glyphs, dungeon);
        obs_.chars = checked_conversion<uint8_t>(chars, dungeon);
        obs_.colors = checked_conversion<uint8_t>(colors, dungeon);
        obs_.specials = checked_conversion<uint8_t>(specials, dungeon);
        obs_.blstats =
            checked_conversion<long>(blstats, { NLE_BLSTATS_SIZE });
        obs_.message = checked_conversion<uint8_t>(message, { 256 });
        obs_.program_state = checked_conversion<int>(
            std::move(program_state), { NLE_PROGRAM_STATE_SIZE });
        obs_.internal =
            checked_conversion<int>(internal, { NLE_INTERNAL_SIZE });
        obs_.inv_glyphs =
            checked_conversion<int16_t>(inv_glyphs, { NLE_INVENTORY_SIZE });
        obs_.inv_letters =
            checked_conversion<uint8_t>(inv_letters, { NLE_INVENTORY_SIZE });
        obs_.inv_oclasses =
            checked_conversion<uint8_t>(inv_oclasses, { NLE_INVENTORY_SIZE });
        obs_.inv_strs = checked_conversion<uint8_t>(
            inv_strs, { NLE_INVENTORY_SIZE, NLE_INVENTORY_STR_LENGTH });
        obs_.screen_descriptions = checked_conversion<uint8_t>(
            screen_descriptions,
            { ROWNO, COLNO - 1, NLE_SCREEN_DESCRIPTION_LENGTH });
        obs_.tty_chars = checked_conversion<uint8_t>(
            tty_chars, { NLE_TERM_LI, NLE_TERM_CO });
        obs_.tty_colors = checked_conversion<int8_t>(
            tty_colors, { NLE_TERM_LI, NLE_TERM_CO });
        obs_.tty_cursor = checked_conversion<uint8_t>(tty_cursor, { 2 });
        obs_.misc = checked_conversion<int32_t>(misc, { NLE_MISC_SIZE });

        py_buffers_ = { std::move(glyphs),
                        std::move(chars),
                        std::move(colors),
                        std::move(specials),
                        std::move(blstats),
                        std::move(message),
                        std::move(program_state),
                        std::move(internal),
                        std::move(inv_glyphs),
                        std::move(inv_letters),
                        std::move(inv_oclasses),
                        std::move(inv_strs),
                        std::move(screen_descriptions),
                        std::move(tty_chars),
                        std::move(tty_colors),
                        std::move(tty_cursor),
                        std::move(misc) };
    }

    void
    close()
    {
        if (nle_) {
            nle_end(nle_);
            nle_ = nullptr;
        }
    }

    void
    set_initial_seeds(unsigned long core, unsigned long disp, bool reseed)
    {
#ifdef NLE_ALLOW_SEEDING
        seed_init_.seeds[0] = core;
        seed_init_.seeds[1] = disp;
        seed_init_.reseed = reseed;
        use_seed_init = true;
#else
        throw std::runtime_error("Seeding not enabled");
#endif
    }

    void
    set_seeds(unsigned long core, unsigned long disp, bool reseed)
    {
#ifdef NLE_ALLOW_SEEDING
        if (!nle_)
            throw std::runtime_error("set_seed called without reset()");
        nle_set_seed(nle_, core, disp, reseed);
#else
        throw std::runtime_error("Seeding not enabled");
#endif
    }

    std::tuple<unsigned long, unsigned long, bool>
    get_seeds()
    {
#ifdef NLE_ALLOW_SEEDING
        if (!nle_)
            throw std::runtime_error("get_seed called without reset()");
        std::tuple<unsigned long, unsigned long, bool> result;
        char
            reseed; /* NetHack's booleans are not necessarily C++ bools ... */
        nle_get_seed(nle_, &std::get<0>(result), &std::get<1>(result),
                     &reseed);
        std::get<2>(result) = reseed;
        return result;
#else
        throw std::runtime_error("Seeding not enabled");
#endif
    }

    boolean
    in_normal_game()
    {
        return obs_.in_normal_game;
    }

    game_end_types
    how_done()
    {
        return static_cast<game_end_types>(obs_.how_done);
    }

  private:
    void
    reset(FILE *ttyrec)
    {
        if (!nle_) {
            nle_ = nle_start(dlpath_.c_str(), &obs_,
                             ttyrec ? ttyrec : ttyrec_.get(),
                             use_seed_init ? &seed_init_ : nullptr);
        } else
            nle_reset(nle_, &obs_, ttyrec,
                      use_seed_init ? &seed_init_ : nullptr);

        use_seed_init = false;

        if (obs_.done)
            throw std::runtime_error("NetHack done right after reset");
    }

    std::string dlpath_;
    nle_obs obs_;
    std::vector<py::object> py_buffers_;
    nle_seeds_init_t seed_init_;
    bool use_seed_init = false;
    nle_ctx_t *nle_ = nullptr;
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> ttyrec_;
};

PYBIND11_MODULE(_pynethack, m)
{
    m.doc() = "The NetHack Learning Environment";

    py::class_<Nethack>(m, "Nethack")
        .def(py::init<std::string, std::string>(), py::arg("dlpath"),
             py::arg("ttyrec"))
        .def("step", &Nethack::step, py::arg("action"))
        .def("done", &Nethack::done)
        .def("reset", py::overload_cast<>(&Nethack::reset))
        .def("reset", py::overload_cast<std::string>(&Nethack::reset))
        .def("set_buffers", &Nethack::set_buffers,
             py::arg("glyphs") = py::none(), py::arg("chars") = py::none(),
             py::arg("colors") = py::none(), py::arg("specials") = py::none(),
             py::arg("blstats") = py::none(), py::arg("message") = py::none(),
             py::arg("program_state") = py::none(),
             py::arg("internal") = py::none(),
             py::arg("inv_glyphs") = py::none(),
             py::arg("inv_letters") = py::none(),
             py::arg("inv_oclasses") = py::none(),
             py::arg("inv_strs") = py::none(),
             py::arg("screen_descriptions") = py::none(),
             py::arg("tty_chars") = py::none(),
             py::arg("tty_colors") = py::none(),
             py::arg("tty_cursor") = py::none(), py::arg("misc") = py::none())
        .def("close", &Nethack::close)
        .def("set_initial_seeds", &Nethack::set_initial_seeds)
        .def("set_seeds", &Nethack::set_seeds)
        .def("get_seeds", &Nethack::get_seeds)
        .def("in_normal_game", &Nethack::in_normal_game)
        .def("how_done", &Nethack::how_done);

    py::module mn = m.def_submodule(
        "nethack", "Collection of NetHack constants and functions");

    /* NLE specific constants. */
    mn.attr("NLE_MESSAGE_SIZE") = py::int_(NLE_MESSAGE_SIZE);
    mn.attr("NLE_BLSTATS_SIZE") = py::int_(NLE_BLSTATS_SIZE);
    mn.attr("NLE_PROGRAM_STATE_SIZE") = py::int_(NLE_PROGRAM_STATE_SIZE);
    mn.attr("NLE_INTERNAL_SIZE") = py::int_(NLE_INTERNAL_SIZE);
    mn.attr("NLE_MISC_SIZE") = py::int_(NLE_MISC_SIZE);
    mn.attr("NLE_INVENTORY_SIZE") = py::int_(NLE_INVENTORY_SIZE);
    mn.attr("NLE_INVENTORY_STR_LENGTH") = py::int_(NLE_INVENTORY_STR_LENGTH);
    mn.attr("NLE_SCREEN_DESCRIPTION_LENGTH") =
        py::int_(NLE_SCREEN_DESCRIPTION_LENGTH);

    mn.attr("NLE_ALLOW_SEEDING") =
#ifdef NLE_ALLOW_SEEDING
        true;
#else
        false;
#endif

    /* NetHack constants. */
    mn.attr("ROWNO") = py::int_(ROWNO);
    mn.attr("COLNO") = py::int_(COLNO);
    mn.attr("NLE_TERM_LI") = py::int_(NLE_TERM_LI);
    mn.attr("NLE_TERM_CO") = py::int_(NLE_TERM_CO);

    mn.attr("NHW_MESSAGE") = py::int_(NHW_MESSAGE);
    mn.attr("NHW_STATUS") = py::int_(NHW_STATUS);
    mn.attr("NHW_MAP") = py::int_(NHW_MAP);
    mn.attr("NHW_MENU") = py::int_(NHW_MENU);
    mn.attr("NHW_TEXT") = py::int_(NHW_TEXT);

    // Cannot include wintty.h as it redefines putc etc.
    // MAXWIN is #defined as 20 there.
    mn.attr("MAXWIN") = py::int_(20);

    mn.attr("NUMMONS") = py::int_(NUMMONS);
    mn.attr("NUM_OBJECTS") = py::int_(NUM_OBJECTS);

    // Glyph array offsets. This is what the glyph_is_* functions
    // are based on, see display.h.
    mn.attr("GLYPH_MON_OFF") = py::int_(GLYPH_MON_OFF);
    mn.attr("GLYPH_PET_OFF") = py::int_(GLYPH_PET_OFF);
    mn.attr("GLYPH_INVIS_OFF") = py::int_(GLYPH_INVIS_OFF);
    mn.attr("GLYPH_DETECT_OFF") = py::int_(GLYPH_DETECT_OFF);
    mn.attr("GLYPH_BODY_OFF") = py::int_(GLYPH_BODY_OFF);
    mn.attr("GLYPH_RIDDEN_OFF") = py::int_(GLYPH_RIDDEN_OFF);
    mn.attr("GLYPH_OBJ_OFF") = py::int_(GLYPH_OBJ_OFF);
    mn.attr("GLYPH_CMAP_OFF") = py::int_(GLYPH_CMAP_OFF);
    mn.attr("GLYPH_EXPLODE_OFF") = py::int_(GLYPH_EXPLODE_OFF);
    mn.attr("GLYPH_ZAP_OFF") = py::int_(GLYPH_ZAP_OFF);
    mn.attr("GLYPH_SWALLOW_OFF") = py::int_(GLYPH_SWALLOW_OFF);
    mn.attr("GLYPH_WARNING_OFF") = py::int_(GLYPH_WARNING_OFF);
    mn.attr("GLYPH_STATUE_OFF") = py::int_(GLYPH_STATUE_OFF);
    mn.attr("MAX_GLYPH") = py::int_(MAX_GLYPH);

    mn.attr("NO_GLYPH") = py::int_(NO_GLYPH);
    mn.attr("GLYPH_INVISIBLE") = py::int_(GLYPH_INVISIBLE);

    mn.attr("MAXEXPCHARS") = py::int_(MAXEXPCHARS);
    mn.attr("MAXPCHARS") = py::int_(static_cast<int>(MAXPCHARS));
    mn.attr("EXPL_MAX") = py::int_(static_cast<int>(EXPL_MAX));
    mn.attr("NUM_ZAP") = py::int_(static_cast<int>(NUM_ZAP));
    mn.attr("WARNCOUNT") = py::int_(static_cast<int>(WARNCOUNT));

    // From objclass.h
    mn.attr("RANDOM_CLASS") = py::int_(static_cast<int>(
        RANDOM_CLASS)); /* used for generating random objects */
    mn.attr("ILLOBJ_CLASS") = py::int_(static_cast<int>(ILLOBJ_CLASS));
    mn.attr("WEAPON_CLASS") = py::int_(static_cast<int>(WEAPON_CLASS));
    mn.attr("ARMOR_CLASS") = py::int_(static_cast<int>(ARMOR_CLASS));
    mn.attr("RING_CLASS") = py::int_(static_cast<int>(RING_CLASS));
    mn.attr("AMULET_CLASS") = py::int_(static_cast<int>(AMULET_CLASS));
    mn.attr("TOOL_CLASS") = py::int_(static_cast<int>(TOOL_CLASS));
    mn.attr("FOOD_CLASS") = py::int_(static_cast<int>(FOOD_CLASS));
    mn.attr("POTION_CLASS") = py::int_(static_cast<int>(POTION_CLASS));
    mn.attr("SCROLL_CLASS") = py::int_(static_cast<int>(SCROLL_CLASS));
    mn.attr("SPBOOK_CLASS") =
        py::int_(static_cast<int>(SPBOOK_CLASS)); /* actually SPELL-book */
    mn.attr("WAND_CLASS") = py::int_(static_cast<int>(WAND_CLASS));
    mn.attr("COIN_CLASS") = py::int_(static_cast<int>(COIN_CLASS));
    mn.attr("GEM_CLASS") = py::int_(static_cast<int>(GEM_CLASS));
    mn.attr("ROCK_CLASS") = py::int_(static_cast<int>(ROCK_CLASS));
    mn.attr("BALL_CLASS") = py::int_(static_cast<int>(BALL_CLASS));
    mn.attr("CHAIN_CLASS") = py::int_(static_cast<int>(CHAIN_CLASS));
    mn.attr("VENOM_CLASS") = py::int_(static_cast<int>(VENOM_CLASS));
    mn.attr("MAXOCLASSES") = py::int_(static_cast<int>(MAXOCLASSES));

    // From monsym.h.
    mn.attr("MAXMCLASSES") = py::int_(static_cast<int>(MAXMCLASSES));

    // game_end_types from hack.h (used in end.c)
    py::enum_<game_end_types>(mn, "game_end_types",
                              "This is the way the game ends.")
        .value("DIED", DIED)
        .value("CHOKING", CHOKING)
        .value("POISONING", POISONING)
        .value("STARVING", STARVING)
        .value("DROWNING", DROWNING)
        .value("BURNING", BURNING)
        .value("DISSOLVED", DISSOLVED)
        .value("CRUSHING", CRUSHING)
        .value("STONING", STONING)
        .value("TURNED_SLIME", TURNED_SLIME)
        .value("GENOCIDED", GENOCIDED)
        .value("PANICKED", PANICKED)
        .value("TRICKED", TRICKED)
        .value("QUIT", QUIT)
        .value("ESCAPED", ESCAPED)
        .value("ASCENDED", ASCENDED)
        .export_values();

    // "Special" mapglyph
    mn.attr("MG_CORPSE") = py::int_(MG_CORPSE);
    mn.attr("MG_INVIS") = py::int_(MG_INVIS);
    mn.attr("MG_DETECT") = py::int_(MG_DETECT);
    mn.attr("MG_PET") = py::int_(MG_PET);
    mn.attr("MG_RIDDEN") = py::int_(MG_RIDDEN);
    mn.attr("MG_STATUE") = py::int_(MG_STATUE);
    mn.attr("MG_OBJPILE") =
        py::int_(MG_OBJPILE); /* more than one stack of objects */
    mn.attr("MG_BW_LAVA") = py::int_(MG_BW_LAVA); /* 'black & white lava' */

    // Expose macros as Python functions.
    mn.def("glyph_is_monster",
           [](int glyph) { return glyph_is_monster(glyph); });
    mn.def("glyph_is_normal_monster",
           [](int glyph) { return glyph_is_normal_monster(glyph); });
    mn.def("glyph_is_pet", [](int glyph) { return glyph_is_pet(glyph); });
    mn.def("glyph_is_body", [](int glyph) { return glyph_is_body(glyph); });
    mn.def("glyph_is_statue",
           [](int glyph) { return glyph_is_statue(glyph); });
    mn.def("glyph_is_ridden_monster",
           [](int glyph) { return glyph_is_ridden_monster(glyph); });
    mn.def("glyph_is_detected_monster",
           [](int glyph) { return glyph_is_detected_monster(glyph); });
    mn.def("glyph_is_invisible",
           [](int glyph) { return glyph_is_invisible(glyph); });
    mn.def("glyph_is_normal_object",
           [](int glyph) { return glyph_is_normal_object(glyph); });
    mn.def("glyph_is_object",
           [](int glyph) { return glyph_is_object(glyph); });
    mn.def("glyph_is_trap", [](int glyph) { return glyph_is_trap(glyph); });
    mn.def("glyph_is_cmap", [](int glyph) { return glyph_is_cmap(glyph); });
    mn.def("glyph_is_swallow",
           [](int glyph) { return glyph_is_swallow(glyph); });
    mn.def("glyph_is_warning",
           [](int glyph) { return glyph_is_warning(glyph); });

    py::class_<permonst>(mn, "permonst", "The permonst struct.")
        .def(
            "__init__",
            // See https://github.com/pybind/pybind11/issues/2394
            [](py::detail::value_and_holder &v_h, int index) {
                if (index < 0 || index >= NUMMONS)
                    throw std::out_of_range(
                        "Index should be between 0 and NUMMONS ("
                        + std::to_string(NUMMONS) + ") but got "
                        + std::to_string(index));
                v_h.value_ptr() = &mons[index];
                v_h.inst->owned = false;
                v_h.set_holder_constructed(true);
            },
            py::detail::is_new_style_constructor())
        .def_readonly("mname", &permonst::mname)   /* full name */
        .def_readonly("mlet", &permonst::mlet)     /* symbol */
        .def_readonly("mlevel", &permonst::mlevel) /* base monster level */
        .def_readonly("mmove", &permonst::mmove)   /* move speed */
        .def_readonly("ac", &permonst::ac)         /* (base) armor class */
        .def_readonly("mr", &permonst::mr) /* (base) magic resistance */
        // .def_readonly("maligntyp", &permonst::maligntyp) /* basic
        // monster alignment */
        .def_readonly("geno", &permonst::geno) /* creation/geno mask value */
        // .def_readonly("mattk", &permonst::mattk) /* attacks matrix
        // */
        .def_readonly("cwt", &permonst::cwt) /* weight of corpse */
        .def_readonly("cnutrit",
                      &permonst::cnutrit) /* its nutritional value */
        .def_readonly("msound",
                      &permonst::msound)         /* noise it makes (6 bits) */
        .def_readonly("msize", &permonst::msize) /* physical size (3 bits) */
        .def_readonly("mresists", &permonst::mresists) /* resistances */
        .def_readonly("mconveys",
                      &permonst::mconveys)           /* conveyed by eating */
        .def_readonly("mflags1", &permonst::mflags1) /* boolean bitflags */
        .def_readonly("mflags2",
                      &permonst::mflags2) /* more boolean bitflags */
        .def_readonly("mflags3",
                      &permonst::mflags3) /* yet more boolean bitflags */
#ifdef TEXTCOLOR
        .def_readonly("mcolor", &permonst::mcolor) /* color to use */
#endif
        ;

    py::class_<class_sym>(mn, "class_sym")
        .def_static(
            "from_mlet",
            [](char let) -> const class_sym * {
                if (let < 0 || let >= MAXMCLASSES)
                    throw std::out_of_range(
                        "Argument should be between 0 and MAXMCLASSES ("
                        + std::to_string(MAXMCLASSES) + ") but got "
                        + std::to_string(let));
                return &def_monsyms[let];
            },
            py::return_value_policy::reference)
        .def_readonly("sym", &class_sym::sym)
        .def_readonly("name", &class_sym::name)
        .def_readonly("explain", &class_sym::explain)
        .def("__repr__", [](const class_sym &cs) {
            return "<nethack.class_sym sym='" + std::string(1, cs.sym)
                   + "' explain='" + std::string(cs.explain) + "'>";
        });

    mn.def("glyph_to_mon", [](int glyph) { return glyph_to_mon(glyph); });
    mn.def("glyph_to_obj", [](int glyph) { return glyph_to_obj(glyph); });
    mn.def("glyph_to_trap", [](int glyph) { return glyph_to_trap(glyph); });
    mn.def("glyph_to_cmap", [](int glyph) { return glyph_to_cmap(glyph); });
    mn.def("glyph_to_swallow",
           [](int glyph) { return glyph_to_swallow(glyph); });
    mn.def("glyph_to_warning",
           [](int glyph) { return glyph_to_warning(glyph); });

    py::class_<objclass>(
        mn, "objclass",
        "The objclass struct.\n\n"
        "All fields are constant and don't reflect user changes.")
        .def(
            "__init__",
            // See https://github.com/pybind/pybind11/issues/2394
            [](py::detail::value_and_holder &v_h, int i) {
                if (i < 0 || i >= NUM_OBJECTS)
                    throw std::out_of_range(
                        "Index should be between 0 and NUM_OBJECTS ("
                        + std::to_string(NUM_OBJECTS) + ") but got "
                        + std::to_string(i));

                /* Initialize. Cannot depend on o_init.c as it pulls
                 * in all kinds of other code. Instead, do what
                 * makedefs.c does at set it here.
                 * Alternative: Get the pointer from the game itself?
                 * Dangerous!
                 */
                objects[i].oc_name_idx = objects[i].oc_descr_idx = i;

                v_h.value_ptr() = &objects[i];
                v_h.inst->owned = false;
                v_h.set_holder_constructed(true);
            },
            py::detail::is_new_style_constructor())
        .def_readonly("oc_name_idx",
                      &objclass::oc_name_idx) /* index of actual name */
        .def_readonly(
            "oc_descr_idx",
            &objclass::oc_descr_idx) /* description when name unknown */
        .def_readonly(
            "oc_oprop",
            &objclass::oc_oprop) /* property (invis, &c.) conveyed */
        .def_readonly(
            "oc_class",
            &objclass::oc_class) /* object class (enum obj_class_types) */
        .def_readonly(
            "oc_delay",
            &objclass::oc_delay) /* delay when using such an object */
        .def_readonly("oc_color",
                      &objclass::oc_color) /* color of the object */

        .def_readonly("oc_prob",
                      &objclass::oc_prob) /* probability, used in mkobj() */
        .def_readonly("oc_weight",
                      &objclass::oc_weight) /* encumbrance (1 cn = 0.1 lb.) */
        .def_readonly("oc_cost", &objclass::oc_cost) /* base cost in shops */
        /* And much more, see objclass.h. */;

    mn.def("OBJ_NAME", [](const objclass &obj) { return OBJ_NAME(obj); });
    mn.def("OBJ_DESCR", [](const objclass &obj) { return OBJ_DESCR(obj); });

    py::class_<objdescr>(mn, "objdescr")
        .def_static(
            "from_idx",
            [](int idx) -> const objdescr * {
                if (idx < 0 || idx >= NUM_OBJECTS)
                    throw std::out_of_range(
                        "Argument should be between 0 and NUM_OBJECTS ("
                        + std::to_string(NUM_OBJECTS) + ") but got "
                        + std::to_string(idx));
                return &obj_descr[idx];
            },
            py::return_value_policy::reference)
        .def_readonly("oc_name", &objdescr::oc_name)
        .def_readonly("oc_descr", &objdescr::oc_descr)
        .def("__repr__", [](const objdescr &od) {
            // clang-format doesn't like the _s UDL.
            // clang-format off
            return "<nethack.objdescr oc_name={!r} oc_descr={!r}>"_s
                // clang-format on
                .format(od.oc_name ? py::str(od.oc_name)
                                   : py::object(py::none()),
                        od.oc_descr ? py::str(od.oc_descr)
                                    : py::object(py::none()));
        });
}
