/* Copyright (c) Facebook, Inc. and its affiliates. */
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

// From drawing.c. Needs drawing.o at link time.
// extern const struct class_sym def_monsyms[MAXMCLASSES];

namespace py = pybind11;

class NLE
{
  public:
    NLE() : obs_{ 0, 0, nullptr, nullptr, nullptr }
    {
        obs_.chars = &chars_[0];
        nle_ = nle_start(&obs_);
    }
    ~NLE()
    {
        nle_end(nle_);
    }
    void
    step(int action)
    {
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
        nle_reset(nle_, &obs_);
    }

    char *
    observation()
    {
        return &chars_[0];
    }

  private:
    char chars_[ROWNO * (COLNO - 1)];
    nle_obs obs_;
    nle_ctx_t *nle_;
};

PYBIND11_MODULE(pynle, m)
{
    m.doc() = "The NetHack Learning Environment";

    py::class_<NLE>(m, "NLE")
        .def(py::init<>())
        .def("step", &NLE::step, py::arg("action"))
        .def("done", &NLE::done)
        .def("reset", &NLE::reset)
        .def("observation", [](NLE &self) {
            return py::array(
                py::buffer_info(self.observation(), ROWNO * (COLNO - 1)));
        });

    m.attr("NHW_MESSAGE") = py::int_(NHW_MESSAGE);
    m.attr("NHW_STATUS") = py::int_(NHW_STATUS);
    m.attr("NHW_MAP") = py::int_(NHW_MAP);
    m.attr("NHW_MENU") = py::int_(NHW_MENU);
    m.attr("NHW_TEXT") = py::int_(NHW_TEXT);

    // m.attr("MAXWIN") = py::int_(MAXWIN);

    m.attr("NUMMONS") = py::int_(NUMMONS);

    // Glyph array offsets. This is what the glyph_is_* functions
    // are based on, see display.h.
    m.attr("GLYPH_MON_OFF") = py::int_(GLYPH_MON_OFF);
    m.attr("GLYPH_PET_OFF") = py::int_(GLYPH_PET_OFF);
    m.attr("GLYPH_INVIS_OFF") = py::int_(GLYPH_INVIS_OFF);
    m.attr("GLYPH_DETECT_OFF") = py::int_(GLYPH_DETECT_OFF);
    m.attr("GLYPH_BODY_OFF") = py::int_(GLYPH_BODY_OFF);
    m.attr("GLYPH_RIDDEN_OFF") = py::int_(GLYPH_RIDDEN_OFF);
    m.attr("GLYPH_OBJ_OFF") = py::int_(GLYPH_OBJ_OFF);
    m.attr("GLYPH_CMAP_OFF") = py::int_(GLYPH_CMAP_OFF);
    m.attr("GLYPH_EXPLODE_OFF") = py::int_(GLYPH_EXPLODE_OFF);
    m.attr("GLYPH_ZAP_OFF") = py::int_(GLYPH_ZAP_OFF);
    m.attr("GLYPH_SWALLOW_OFF") = py::int_(GLYPH_SWALLOW_OFF);
    m.attr("GLYPH_WARNING_OFF") = py::int_(GLYPH_WARNING_OFF);
    m.attr("GLYPH_STATUE_OFF") = py::int_(GLYPH_STATUE_OFF);
    m.attr("MAX_GLYPH") = py::int_(MAX_GLYPH);

    m.attr("NO_GLYPH") = py::int_(NO_GLYPH);
    m.attr("GLYPH_INVISIBLE") = py::int_(GLYPH_INVISIBLE);

    m.attr("MAXPCHARS") = py::int_(static_cast<int>(MAXPCHARS));
    m.attr("EXPL_MAX") = py::int_(static_cast<int>(EXPL_MAX));
    m.attr("NUM_ZAP") = py::int_(static_cast<int>(NUM_ZAP));
    m.attr("WARNCOUNT") = py::int_(static_cast<int>(WARNCOUNT));

    // From objclass.h
    m.attr("RANDOM_CLASS") = py::int_(static_cast<int>(
        RANDOM_CLASS)); /* used for generating random objects */
    m.attr("ILLOBJ_CLASS") = py::int_(static_cast<int>(ILLOBJ_CLASS));
    m.attr("WEAPON_CLASS") = py::int_(static_cast<int>(WEAPON_CLASS));
    m.attr("ARMOR_CLASS") = py::int_(static_cast<int>(ARMOR_CLASS));
    m.attr("RING_CLASS") = py::int_(static_cast<int>(RING_CLASS));
    m.attr("AMULET_CLASS") = py::int_(static_cast<int>(AMULET_CLASS));
    m.attr("TOOL_CLASS") = py::int_(static_cast<int>(TOOL_CLASS));
    m.attr("FOOD_CLASS") = py::int_(static_cast<int>(FOOD_CLASS));
    m.attr("POTION_CLASS") = py::int_(static_cast<int>(POTION_CLASS));
    m.attr("SCROLL_CLASS") = py::int_(static_cast<int>(SCROLL_CLASS));
    m.attr("SPBOOK_CLASS") =
        py::int_(static_cast<int>(SPBOOK_CLASS)); /* actually SPELL-book */
    m.attr("WAND_CLASS") = py::int_(static_cast<int>(WAND_CLASS));
    m.attr("COIN_CLASS") = py::int_(static_cast<int>(COIN_CLASS));
    m.attr("GEM_CLASS") = py::int_(static_cast<int>(GEM_CLASS));
    m.attr("ROCK_CLASS") = py::int_(static_cast<int>(ROCK_CLASS));
    m.attr("BALL_CLASS") = py::int_(static_cast<int>(BALL_CLASS));
    m.attr("CHAIN_CLASS") = py::int_(static_cast<int>(CHAIN_CLASS));
    m.attr("VENOM_CLASS") = py::int_(static_cast<int>(VENOM_CLASS));
    m.attr("MAXOCLASSES") = py::int_(static_cast<int>(MAXOCLASSES));

    // "Special" mapglyph
    m.attr("MG_CORPSE") = py::int_(MG_CORPSE);
    m.attr("MG_INVIS") = py::int_(MG_INVIS);
    m.attr("MG_DETECT") = py::int_(MG_DETECT);
    m.attr("MG_PET") = py::int_(MG_PET);
    m.attr("MG_RIDDEN") = py::int_(MG_RIDDEN);
    m.attr("MG_STATUE") = py::int_(MG_STATUE);
    m.attr("MG_OBJPILE") =
        py::int_(MG_OBJPILE); /* more than one stack of objects */
    m.attr("MG_BW_LAVA") = py::int_(MG_BW_LAVA); /* 'black & white lava' */

    // Expose macros as Python functions.
    m.def("glyph_is_monster",
          [](int glyph) { return glyph_is_monster(glyph); });
    m.def("glyph_is_normal_monster",
          [](int glyph) { return glyph_is_normal_monster(glyph); });
    m.def("glyph_is_pet", [](int glyph) { return glyph_is_pet(glyph); });
    m.def("glyph_is_body", [](int glyph) { return glyph_is_body(glyph); });
    m.def("glyph_is_statue",
          [](int glyph) { return glyph_is_statue(glyph); });
    m.def("glyph_is_ridden_monster",
          [](int glyph) { return glyph_is_ridden_monster(glyph); });
    m.def("glyph_is_detected_monster",
          [](int glyph) { return glyph_is_detected_monster(glyph); });
    m.def("glyph_is_invisible",
          [](int glyph) { return glyph_is_invisible(glyph); });
    m.def("glyph_is_normal_object",
          [](int glyph) { return glyph_is_normal_object(glyph); });
    m.def("glyph_is_object",
          [](int glyph) { return glyph_is_object(glyph); });
    m.def("glyph_is_trap", [](int glyph) { return glyph_is_trap(glyph); });
    m.def("glyph_is_cmap", [](int glyph) { return glyph_is_cmap(glyph); });
    m.def("glyph_is_swallow",
          [](int glyph) { return glyph_is_swallow(glyph); });
    m.def("glyph_is_warning",
          [](int glyph) { return glyph_is_warning(glyph); });

    py::class_<permonst>(m, "permonst")
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

    py::class_<class_sym>(m, "class_sym")
        .def_readonly("sym", &class_sym::sym)
        .def_readonly("name", &class_sym::name)
        .def_readonly("explain", &class_sym::explain)
        .def("__repr__", [](const class_sym &cs) {
            return "<nethack.pynle.class_sym sym='" + std::string(1, cs.sym)
                   + "' explain='" + std::string(cs.explain) + "'>";
        });

    /*
    m.def(
        "glyph_to_mon",
        [](int glyph) -> const permonst * {
            return &mons[glyph_to_mon(glyph)];
        },
        py::return_value_policy::reference);

    m.def(
        "mlet_to_class_sym",
        [](char let) -> const class_sym * { return &def_monsyms[let]; },
        py::return_value_policy::reference);
    */
}
