/* Copyright (c) Facebook, Inc. and its affiliates. */
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <stdio.h>
#include <string>
#include <cstring>
#include <unistd.h>
#include <vector>

extern "C" {
#include "hack.h"
}

extern "C" {
#include "wintty.h"
}

extern "C" {
#include "nleobs.h"
}

#define USE_DEBUG_API 0

#if USE_DEBUG_API
#define DEBUG_API(x)    \
    do {                \
        std::cerr << x; \
    } while (0)
#else
#define DEBUG_API(x)
#endif

/*
 * We had to change xwaitforspace() in getline.c to tell the agent in a
 * --More-- situation that enter/return (ironically not necessarily space)
 * is required to continue.
 */
extern bool xwaitingforspace;

/* some hack.h macros. Can be undefined here. */
#undef Invisible
#undef Warning
#undef index
#undef msleep
#undef rindex
#undef wizard
#undef yn

extern unsigned long nle_seeds[];

extern "C" {
extern void *nle_yield(boolean);
extern nle_obs *nle_get_obs();
}

namespace nethack_rl
{
std::deque<std::string> win_proc_calls;
bool in_yn_function = false;
bool in_getlin = false;

class ScopedStack
{
  public:
    ScopedStack(std::deque<std::string> &deque, std::string &&s)
        : deque_(deque)
    {
        deque_.push_back(s);
    }

    ~ScopedStack()
    {
        deque_.pop_back();
    }

  private:
    std::deque<std::string> &deque_;
};

class NetHackRL
{
  public:
    NetHackRL(int &argc, char **argv);

    static void rl_init_nhwindows(int *argc, char **argv);
    static void rl_player_selection();
    static void rl_askname();
    static void rl_get_nh_event();
    static void rl_exit_nhwindows(const char *);
    static void rl_suspend_nhwindows(const char *);
    static void rl_resume_nhwindows();
    static winid rl_create_nhwindow(int type);
    static void rl_clear_nhwindow(winid wid);
    static void rl_display_nhwindow(winid wid, BOOLEAN_P block);
    static void rl_destroy_nhwindow(winid wid);
    static void rl_curs(winid wid, int x, int y);
    static void rl_putstr(winid wid, int attr, const char *text);
    static void rl_display_file(const char *filename, BOOLEAN_P must_exist);
    static void rl_start_menu(winid wid);
    static void rl_add_menu(winid wid, int glyph, const ANY_P *identifier,
                            CHAR_P ch, CHAR_P gch, int attr, const char *str,
                            BOOLEAN_P presel);
    static void rl_end_menu(winid wid, const char *prompt);
    static int rl_select_menu(winid wid, int how, MENU_ITEM_P **menu_list);
    static void rl_update_inventory();
    static void rl_mark_synch();
    static void rl_wait_synch();

    static void rl_cliparound(int x, int y);
    static void rl_print_glyph(winid wid, XCHAR_P x, XCHAR_P y, int glyph,
                               int bkglyph);
    static void rl_raw_print(const char *str);
    static void rl_raw_print_bold(const char *str);
    static int rl_nhgetch();
    static int rl_nh_poskey(int *x, int *y, int *mod);
    static void rl_nhbell();
    static int rl_doprev_message();
    static char rl_yn_function(const char *question, const char *choices,
                               CHAR_P def);
    static void rl_getlin(const char *prompt, char *line);
    static int rl_get_ext_cmd();
    static void rl_number_pad(int);
    static void rl_delay_output();
    static void rl_start_screen();
    static void rl_end_screen();

    static char *rl_getmsghistory(BOOLEAN_P init);
    static void rl_putmsghistory(const char *msg, BOOLEAN_P is_restoring);

    static void rl_outrip(winid wid, int how, time_t when);
    static void rl_status_init();

    static void rl_status_update(int fldidx, genericptr_t ptr, int chg,
                                 int percent, int color,
                                 unsigned long *colormasks);

  private:
    struct rl_menu_item {
        int glyph;           /* character glyph */
        anything identifier; /* user identifier */
        long count;          /* user count */
        std::string str;     /* description string */
        int attr;            /* string attribute */
        boolean selected;    /* TRUE if selected by user */
        char selector;       /* keyboard accelerator */
        char gselector;      /* group accelerator */
    };

    struct rl_window {
        int type;
        std::vector<rl_menu_item> menu_items;
        std::vector<std::string> strings;
    };

    struct rl_inventory_item {
        int glyph;
        // TODO: Don't heap allocate this stuff.
        std::string str;
        char letter;
        char object_class;
        // TODO: Don't heap allocate this stuff.
        std::string object_class_name;
    };

    static std::unique_ptr<NetHackRL> instance;

    // TODO: Don't heap allocate this stuff.
    std::vector<std::unique_ptr<rl_window> > windows_;

    std::array<int16_t, (COLNO - 1) * ROWNO> glyphs_;

    /* Output of mapglyph */
    std::array<uint8_t, (COLNO - 1) * ROWNO> chars_;
    std::array<uint8_t, (COLNO - 1) * ROWNO> colors_;
    std::array<uint8_t, (COLNO - 1) * ROWNO> specials_;

    std::array<char, (COLNO - 1) * ROWNO * NLE_SCREEN_DESCRIPTION_LENGTH> screen_descriptions_;

    void store_glyph(XCHAR_P x, XCHAR_P y, int glyph);
    void store_mapped_glyph(int ch, int color, int special, XCHAR_P x,
                            XCHAR_P y);
    void store_screen_description(XCHAR_P x, XCHAR_P y, int glyph);

    void fill_obs(nle_obs *);
    int getch_method();

    std::array<std::string, MAXBLSTATS> status_;
    long condition_bits_;

    void player_selection_method();
    void status_update_method(int fldidx, genericptr_t ptr, int, int percent,
                              int color, unsigned long *colormasks);

    void putstr_method(winid wid, int attr, const char *str);

    std::vector<rl_inventory_item> inventory_;

    void start_menu_method(winid wid);
    void add_menu_method(winid wid, int glyph, const anything *identifier,
                         char ch, char gch, int attr, const char *str,
                         bool preselected);
    void update_inventory_method();

    winid create_nhwindow_method(int type);
    void clear_nhwindow_method(winid wid);
    void display_nhwindow_method(winid wid, BOOLEAN_P block);
    void destroy_nhwindow_method(winid wid);
};

std::unique_ptr<NetHackRL> NetHackRL::instance =
    std::unique_ptr<NetHackRL>(nullptr);

NetHackRL::NetHackRL(int &argc, char **argv) : glyphs_()
{
    // create base window
    // (done in tty_init_nhwindows before this NetHackRL object got created).
    assert(BASE_WINDOW == 0);
    windows_.emplace_back(new rl_window({ NHW_BASE }));
}

void
NetHackRL::player_selection_method()
{
    windows_[BASE_WINDOW]->strings.clear();
}

void
NetHackRL::fill_obs(nle_obs *obs)
{
    if (obs->program_state) {
        obs->program_state[0] = program_state.gameover;
        obs->program_state[1] = program_state.panicking;
        obs->program_state[2] = program_state.exiting;
        obs->program_state[3] = program_state.in_moveloop;
        obs->program_state[4] = program_state.in_impossible;
        obs->program_state[5] = program_state.something_worth_saving;
        // TODO: Consider adding something_worth_saving.
        // Also consider adding ttyDisplay->inmore ...
    }
    if (obs->internal) {
        // From do.c. sstairs is a potential "special" staircase.
        boolean stairs_down =
            ((u.ux == xdnstair && u.uy == ydnstair)
             || (u.ux == sstairs.sx && u.uy == sstairs.sy && !sstairs.up));

        obs->internal[0] = deepest_lev_reached(false);
        obs->internal[1] = in_yn_function;
        obs->internal[2] = in_getlin;
        obs->internal[3] = xwaitingforspace;
        obs->internal[4] = stairs_down;
        obs->internal[5] = nle_seeds[0]; /* core */
        obs->internal[6] = nle_seeds[1]; /* disp */
        obs->internal[7] = u.uhunger;
    }

    if ((!program_state.something_worth_saving && !program_state.in_moveloop)
        || !iflags.window_inited) {
        // Game not yet started (!something_worth_saving && !in_moveloop -- we
        // need both as something_worth_saving also becomes false in
        // really_done(), but we still want to see the "Do you want..."
        // questions) or windows have already been destroyed. Return zero
        // observations.
        obs->in_normal_game = false;
        if (obs->glyphs)
            std::memset(obs->glyphs, 0, sizeof(int16_t) * glyphs_.size());
        if (obs->chars)
            std::memset(obs->chars, 0, chars_.size());
        if (obs->colors)
            std::memset(obs->colors, 0, colors_.size());
        if (obs->specials)
            std::memset(obs->specials, 0, specials_.size());
        if (obs->message)
            std::memset(obs->message, 0, NLE_MESSAGE_SIZE);
        if (obs->blstats)
            std::memset(obs->blstats, 0, sizeof(long) * NLE_BLSTATS_SIZE);
        if (obs->screen_descriptions)
            std::memset(obs->screen_descriptions, 0,
                        screen_descriptions_.size());
        return;
    }
    obs->in_normal_game = true;

    if (obs->glyphs) {
        std::memcpy(obs->glyphs, glyphs_.data(),
                    sizeof(int16_t) * glyphs_.size());
    }
    if (obs->chars) {
        std::memcpy(obs->chars, chars_.data(), chars_.size());
    }
    if (obs->colors) {
        std::memcpy(obs->colors, colors_.data(), colors_.size());
    }
    if (obs->specials) {
        std::memcpy(obs->specials, specials_.data(), specials_.size());
    }
    if (obs->message) {
        // TODO: This doesn't show anything in situations where there's too
        // many items at one tile, which will get displayed in a new window.

        if (in_yn_function) {
            // Special case. See tty_putstr: yn_function doesn't add to
            // toplines until after that frame is over. Use last string on
            // NHW_MESSAGE instead.
            assert(windows_.size() > WIN_MESSAGE);
            rl_window *win = windows_[WIN_MESSAGE].get();
            assert(win->type == NHW_MESSAGE);
            std::strncpy((char *) &obs->message[0],
                         win->strings.back().c_str(), NLE_MESSAGE_SIZE);
        } else if (ttyDisplay->toplin) {
            // Copy toplines[], see topl.c.
            std::strncpy((char *) &obs->message[0], toplines,
                         NLE_MESSAGE_SIZE);
        } else {
            std::memset(obs->message, 0, NLE_MESSAGE_SIZE);
        }
    }
    if (obs->blstats) {
        // Blstats
        int hitpoints;

        /* See botl.c. */
        int i = Upolyd ? u.mh : u.uhp;
        if (i < 0)
            i = 0;

        hitpoints = min(i, 9999);

        int max_hitpoints;
        i = Upolyd ? u.mhmax : u.uhpmax;
        max_hitpoints = min(i, 9999);

        long blstats[NLE_BLSTATS_SIZE] = {
            /* Cf. botl.c. */
            u.ux - 1,            /* x coordinate, 1 <= ux <= cols */
            u.uy,                /* y coordinate, 0 <= uy < rows */
            ACURRSTR,            /* strength_percentage */
            ACURR(A_STR),        /* strength          */
            ACURR(A_DEX),        /* dexterity         */
            ACURR(A_CON),        /* constitution      */
            ACURR(A_INT),        /* intelligence      */
            ACURR(A_WIS),        /* wisdom            */
            ACURR(A_CHA),        /* charisma          */
            botl_score(),        /* score             */
            hitpoints,           /* hitpoints         */
            max_hitpoints,       /* max_hitpoints     */
            depth(&u.uz),        /* depth             */
            money_cnt(invent),   /* gold              */
            min(u.uen, 9999),    /* energy            */
            min(u.uenmax, 9999), /* max_energy        */
            u.uac,               /* armor_class       */
            Upolyd ? (int) mons[u.umonnum].mlevel : 0, /* monster_level     */
            u.ulevel,                                  /* experience_level  */
            u.uexp,                                    /* experience_points */
            moves,                                     /* time              */
            u.uhs,                                     /* hunger state      */
            near_capacity(),                           /* carrying_capacity */
            u.uz.dnum,                                 /* dungeon number */
            u.uz.dlevel,                               /* level number */
        };

        assert(sizeof(blstats) == NLE_BLSTATS_SIZE * sizeof(long));

        std::memcpy(obs->blstats, &blstats[0], sizeof(blstats));
    }
    if (obs->inv_glyphs) {
        /* This iterates over the inventory_ vector list once per inv
           observation instead of only once. I guess that's fine. */
        int i = 0;
        for (const rl_inventory_item &item : inventory_) {
            obs->inv_glyphs[i++] = item.glyph;
        }
        for (; i < NLE_INVENTORY_SIZE; ++i) {
            obs->inv_glyphs[i] = NO_GLYPH;
        }
    }
    if (obs->inv_strs) {
        int i = 0;
        for (const rl_inventory_item &item : inventory_) {
            int j = 0;
            for (int size = item.str.size(); j < size; ++j) {
                obs->inv_strs[i++] = item.str[j];
            }
            for (; j < NLE_INVENTORY_STR_LENGTH; ++j) {
                obs->inv_strs[i++] = 0;
            }
        }
        for (; i < NLE_INVENTORY_SIZE * NLE_INVENTORY_STR_LENGTH; ++i) {
            obs->inv_strs[i] = 0;
        }
    }
    if (obs->inv_letters) {
        int i = 0;
        for (const rl_inventory_item &item : inventory_) {
            obs->inv_letters[i++] = item.letter;
        }
        for (; i < NLE_INVENTORY_SIZE; ++i) {
            obs->inv_letters[i] = 0;
        }
    }
    if (obs->inv_oclasses) {
        int i = 0;
        for (const rl_inventory_item &item : inventory_) {
            obs->inv_oclasses[i++] = item.object_class;
        }
        for (; i < NLE_INVENTORY_SIZE; ++i) {
            obs->inv_oclasses[i] = MAXOCLASSES;
        }
    }
    if (obs->screen_descriptions) {
        memcpy(obs->screen_descriptions, &screen_descriptions_, screen_descriptions_.size());
    }
}

int
NetHackRL::getch_method()
{
    fill_obs(nle_get_obs());
    int i = ((nle_obs *) nle_yield(TRUE))->action;

    /* NOT calling tty_nhgetch() but instead getting the input from
       the context switch. No stdin required. The following code is from
       tty_nhgetch. */
    if (WIN_MESSAGE != WIN_ERR && wins[WIN_MESSAGE])
        wins[WIN_MESSAGE]->flags &= ~WIN_STOP;
    if (!i)
        i = '\033'; /* map NUL to ESC since nethack doesn't expect NUL */
    else if (i == EOF)
        i = '\033'; /* same for EOF */
    if (ttyDisplay && ttyDisplay->toplin == 1)
        ttyDisplay->toplin = 2;
    DEBUG_API("getch_method: action=" << i << ", xwaitingforspace="
                                      << xwaitingforspace << std::endl);
    return i;
}

void
NetHackRL::update_inventory_method()
{
    /* We cannot simply call display_inventory() as window.doc suggests,
       since we want to also use the tty window proc and we don't want the
       inventory to pop up whenever it changed. Instead, we keep our inventory
       list up to date via the following code adopted from display_pickinv
       in invent.c */

    struct obj *otmp;
    inventory_.clear();

    for (otmp = invent; otmp; otmp = otmp->nobj) {
        inventory_.emplace_back(
            rl_inventory_item{ obj_to_glyph(otmp, rn2_on_display_rng),
                               doname(otmp), otmp->invlet, otmp->oclass,
                               let_to_name(otmp->oclass, false, false) });
    }
}

void
NetHackRL::store_glyph(XCHAR_P x, XCHAR_P y, int glyph)
{
    // 1 <= x < cols, 0 <= y < rows (!)
    size_t i = (x - 1) % (COLNO - 1);
    size_t j = y % ROWNO;
    size_t offset = j * (COLNO - 1) + i;

    // TODO: Glyphs might be taken from gbuf[y][x].glyph.
    glyphs_[offset] = glyph;
}

void
NetHackRL::store_mapped_glyph(int ch, int color, int special, XCHAR_P x,
                              XCHAR_P y)
{
    // 1 <= x < cols, 0 <= y < rows (!)
    size_t i = (x - 1) % (COLNO - 1);
    size_t j = y % ROWNO;
    size_t offset = j * (COLNO - 1) + i;

    chars_[offset] = ch;
    colors_[offset] = color;
    specials_[offset] = special;
}

void
NetHackRL::store_screen_description(XCHAR_P x, XCHAR_P y, int glyph)
{
    // 1 <= x < cols, 0 <= y < rows (!)
    size_t i = (x - 1) % (COLNO - 1);
    size_t j = y % ROWNO;
    size_t offset = j * (COLNO - 1) + i;
    size_t start = offset * NLE_SCREEN_DESCRIPTION_LENGTH;

    // see code in src/do_name.c:538 auto_describe
    coord cc;
    int sym = 0;
    char tmpbuf[BUFSZ];
    const char *firstmatch = "unknown";

    cc.x = x;
    cc.y = y;

    if (do_screen_description(cc, TRUE, sym, tmpbuf, &firstmatch,
                              (struct permonst **) 0)) {
        strncpy((char *) &screen_descriptions_ + start, firstmatch, NLE_SCREEN_DESCRIPTION_LENGTH);
    } else {
        strncpy((char *) &screen_descriptions_ + start, "", NLE_SCREEN_DESCRIPTION_LENGTH);
    }
}

void
NetHackRL::status_update_method(int fldidx, genericptr_t ptr, int,
                                int percent, int color,
                                unsigned long *colormasks)
{
    if ((fldidx < BL_RESET) || (fldidx >= MAXBLSTATS))
        return;

    // Needs to be kept in sync with the switch statement in rl_status_update.
    if (fldidx == BL_FLUSH || fldidx == BL_RESET)
        return;
    else if (fldidx == BL_CONDITION) {
        long *condptr = (long *) ptr;
        condition_bits_ = *condptr;
        return;
    }

    char *text = (char *) ptr;
    std::string status(text);
    if (fldidx == BL_GOLD) {
        // Handle gold glyph.
        char buf[BUFSZ];
        status = decode_mixed(buf, text);
    }
    status_[fldidx] = status;
}

void
NetHackRL::putstr_method(winid wid, int attr, const char *str)
{
    DEBUG_API("About to set strings on " << wid << std::endl);
    windows_[wid]->strings.push_back(str);
}

winid
NetHackRL::create_nhwindow_method(int type)
{
    std::string window_type;
    switch (type) {
    case NHW_MAP:
        window_type = "map";
        break;
    case NHW_MESSAGE:
        window_type = "message";
        break;
    case NHW_STATUS:
        window_type = "status";
        break;
    case NHW_MENU:
        window_type = "menu";
        break;
    case NHW_TEXT:
        window_type = "text";
        break;
    }

    DEBUG_API("rl_create_nhwindow(type=" << window_type << ")");
    ScopedStack s(win_proc_calls, "create_nhwindow");

    winid wid = tty_create_nhwindow(type);
    DEBUG_API(": wid == " << wid << std::endl);

    windows_.resize(wid + 1);
    assert(!windows_[wid]);

    DEBUG_API("ABOUT TO RESET " << wid << std::endl;);

    windows_[wid].reset(new rl_window{ type });
    return wid;
}

void
NetHackRL::clear_nhwindow_method(winid wid)
{
    auto &rl_win = windows_[wid];
    rl_win->menu_items.clear();
    rl_win->strings.clear();

    if (wid == WIN_MAP) {
        glyphs_.fill(0);
        chars_.fill(' ');
        colors_.fill(0);
        specials_.fill(0);
        if (nle_get_obs()->screen_descriptions) {
            screen_descriptions_.fill(0);
        }
    }

    DEBUG_API("rl_clear_nhwindow(wid=" << wid << ")" << std::endl);
    tty_clear_nhwindow(wid);
}

void
NetHackRL::display_nhwindow_method(winid wid, BOOLEAN_P block)
{
    DEBUG_API("rl_display_nhwindow(wid=" << wid << ", block=" << block << ")"
                                         << std::endl);

    tty_display_nhwindow(wid, block);
}

void
NetHackRL::destroy_nhwindow_method(winid wid)
{
    DEBUG_API("rl_destroy_nhwindow(wid=" << wid << ")" << std::endl);
    windows_[wid].reset(nullptr);
    tty_destroy_nhwindow(wid);
}

void
NetHackRL::start_menu_method(winid wid)
{
    DEBUG_API("rl_start_menu(wid=" << wid << ")" << std::endl);
    tty_start_menu(wid);
    windows_[wid]->menu_items.clear();
}

void
NetHackRL::add_menu_method(
    winid wid,                  /* window to use, must be of type NHW_MENU */
    int glyph,                  /* glyph to display with item (not used) */
    const anything *identifier, /* what to return if selected */
    char ch,                    /* keyboard accelerator (0 = pick our own) */
    char gch,                   /* group accelerator (0 = no group) */
    int attr,                   /* attribute for string (like putstr()) */
    const char *str,            /* menu string */
    bool preselected            /* item is marked as selected */
    )
{
    DEBUG_API("rl_add_menu" << std::endl);
    tty_add_menu(wid, glyph, identifier, ch, gch, attr, str, preselected);

    /* We just add the menu item here. One problem with this method is that
       we won't see any updates happening during tty_select_menu. We could
       try to inspect tty's own menu items instead? */

    windows_[wid]->menu_items.emplace_back(rl_menu_item{
        glyph, *identifier, -1L, str, attr, preselected, ch, gch });
}

void
NetHackRL::rl_init_nhwindows(int *argc, char **argv)
{
    DEBUG_API("rl_init_nhwindows" << std::endl);
    ScopedStack s(win_proc_calls, "init_nhwindows");
    tty_init_nhwindows(argc, argv);
    instance = std::make_unique<NetHackRL>(*argc, argv);
}

void
NetHackRL::rl_player_selection()
{
    DEBUG_API("rl_player_selection" << std::endl);
    ScopedStack s(win_proc_calls, "player_selection");
    tty_player_selection();
    instance->player_selection_method();
}

void
NetHackRL::rl_askname()
{
    DEBUG_API("rl_askname" << std::endl);
    ScopedStack s(win_proc_calls, "askname");
    tty_askname();
}

void
NetHackRL::rl_get_nh_event()
{
    DEBUG_API("rl_get_nh_event" << std::endl);
    ScopedStack s(win_proc_calls, "get_nh_event");
    tty_get_nh_event();
}

void
NetHackRL::rl_exit_nhwindows(const char *c)
{
    DEBUG_API("rl_exit_nhwindows" << std::endl);
    ScopedStack s(win_proc_calls, "exit_nhwindows");
    instance.reset(nullptr);
    tty_exit_nhwindows(c);
}

void
NetHackRL::rl_suspend_nhwindows(const char *c)
{
    DEBUG_API("rl_suspend_nhwindows" << std::endl);
    ScopedStack s(win_proc_calls, "suspend_nhwindows");
    tty_suspend_nhwindows(c);
}

void
NetHackRL::rl_resume_nhwindows()
{
    DEBUG_API("rl_resume_nhwindows" << std::endl);
    ScopedStack s(win_proc_calls, "resume_nhwindows");
    tty_resume_nhwindows();
}

winid
NetHackRL::rl_create_nhwindow(int type)
{
    // win_proc_calls code happens in method.
    return instance->create_nhwindow_method(type);
}

void
NetHackRL::rl_clear_nhwindow(winid wid)
{
    ScopedStack s(win_proc_calls, "clear_nhwindow");
    instance->clear_nhwindow_method(wid);
}

/* display_nhwindow(window, boolean blocking)
                -- Display the window on the screen.  If there is data
                   pending for output in that window, it should be sent.
                   If blocking is TRUE, display_nhwindow() will not
                   return until the data has been displayed on the screen,
                   and acknowledged by the user where appropriate.
                -- All calls are blocking in the tty window-port.
                -- Calling display_nhwindow(WIN_MESSAGE,???) will do a
                   --more--, if necessary, in the tty window-port. */
void
NetHackRL::rl_display_nhwindow(winid wid, BOOLEAN_P block)
{
    ScopedStack s(win_proc_calls, "display_nhwindow");
    instance->display_nhwindow_method(wid, block);
}

void
NetHackRL::rl_destroy_nhwindow(winid wid)
{
    ScopedStack s(win_proc_calls, "destroy_nhwindow");
    instance->destroy_nhwindow_method(wid);
}

void
NetHackRL::rl_curs(winid wid, int x, int y)
{
    DEBUG_API("rl_curs(wid=" << wid << ", x=" << x << ", y=" << y << ")"
                             << std::endl);
    ScopedStack s(win_proc_calls, "curs");
    DEBUG_API("rl_curs for window id " << wid << std::endl);
    tty_curs(wid, x, y);
}

void
NetHackRL::rl_putstr(winid wid, int attr, const char *text)
{
    DEBUG_API("rl_putstr(wid=" << wid << ", attr=" << attr
                               << ", text=" << text << ")" << std::endl);
    ScopedStack s(win_proc_calls, "putstr");
    instance->putstr_method(wid, attr, text);
    tty_putstr(wid, attr, text);
}

void
NetHackRL::rl_display_file(const char *filename, BOOLEAN_P must_exist)
{
    DEBUG_API("rl_display_file" << std::endl);
    ScopedStack s(win_proc_calls, "display_file");
    tty_display_file(filename, must_exist);
}

void
NetHackRL::rl_start_menu(winid wid)
{
    ScopedStack s(win_proc_calls, "start_menu");
    instance->start_menu_method(wid);
}

void
NetHackRL::rl_add_menu(winid wid, int glyph, const ANY_P *identifier,
                       CHAR_P ch, CHAR_P gch, int attr, const char *str,
                       BOOLEAN_P presel)
{
    ScopedStack s(win_proc_calls, "add_menu");
    instance->add_menu_method(wid, glyph, identifier, ch, gch, attr, str,
                              presel);
}

void
NetHackRL::rl_end_menu(winid wid, const char *prompt)
{
    DEBUG_API("rl_end_menu" << std::endl);
    ScopedStack s(win_proc_calls, "end_menu");
    tty_end_menu(wid, prompt);
}

int
NetHackRL::rl_select_menu(winid wid, int how, MENU_ITEM_P **menu_list)
{
    DEBUG_API("rl_select_menu");
    ScopedStack s(win_proc_calls, "select_menu");
    int response = tty_select_menu(wid, how, menu_list);
    DEBUG_API(" : " << response << std::endl);
    return response;
}

void
NetHackRL::rl_update_inventory()
{
    DEBUG_API("rl_update_inventory" << std::endl);
    ScopedStack s(win_proc_calls, "update_inventory");
    instance->update_inventory_method();
}

void
NetHackRL::rl_mark_synch()
{
    DEBUG_API("rl_mark_synch" << std::endl);
    ScopedStack s(win_proc_calls, "mark_synch");
    tty_mark_synch();
}

void
NetHackRL::rl_wait_synch()
{
    DEBUG_API("rl_wait_synch" << std::endl);
    ScopedStack s(win_proc_calls, "wait_synch");
    tty_wait_synch();
}

void
NetHackRL::rl_cliparound(int x, int y)
{
#ifdef CLIPPING
    tty_cliparound(x, y);
#endif
}

/* print_glyph(window, x, y, glyph, bkglyph)
                -- Print the glyph at (x,y) on the given window.  Glyphs are
                   integers at the interface, mapped to whatever the window-
                   port wants (symbol, font, color, attributes, ...there's
                   a 1-1 map between glyphs and distinct things on the map).
                -- bkglyph is a background glyph for potential use by some
                   graphical or tiled environments to allow the depiction
                   to fall against a background consistent with the grid
                   around x,y. If bkglyph is NO_GLYPH, then the parameter
                   should be ignored (do nothing with it). */
void
NetHackRL::rl_print_glyph(winid wid, XCHAR_P x, XCHAR_P y, int glyph,
                          int bkglyph)
{
    int ch;
    int color;
    unsigned special;

    (void) mapglyph(glyph, &ch, &color, &special, x, y, 0);
#if USE_DEBUG_API
    DEBUG_API("rl_print_glyph(wid=" << wid << ", x=" << x << ", y=" << y
                                    << ", glyph=(ch='" << (char) ch
                                    << "', color=" << color
                                    << ", special=" << special);
    int bch;
    int bcolor;
    unsigned bspecial;
    (void) mapglyph(bkglyph, &bch, &bcolor, &bspecial, x, y, 0);
    DEBUG_API("), bkglyph=(ch='" << (char) bch << "', color=" << bcolor
                                 << ", special=" << bspecial << ")"
                                 << std::endl);
#endif

    // No win_proc_calls entry here.
    if (wid == WIN_MAP) {
        instance->store_glyph(x, y, glyph);
        instance->store_mapped_glyph(ch, color, special, x, y);
        if (nle_get_obs()->screen_descriptions) {
            instance->store_screen_description(x, y, glyph);
        }
    } else {
        DEBUG_API("Window id is " << wid << ". This shouldn't happen."
                                  << std::endl);
    }

    tty_print_glyph(wid, x, y, glyph, bkglyph);
}
void
NetHackRL::rl_raw_print(const char *str)
{
    DEBUG_API("rl_raw_print" << std::endl);
    ScopedStack s(win_proc_calls, "raw_print");
    /* Not calling tty_raw_print(str); here or below as that
       uses puts/fputs. */
    xputs(str);
    putchar('\n');
    fflush(stdout);
}

void
NetHackRL::rl_raw_print_bold(const char *str)
{
    DEBUG_API("rl_raw_print_bold" << std::endl);
    ScopedStack s(win_proc_calls, "raw_bold_print");
    /* Not calling tty_raw_print_bold(str);, so above. */
    xputs(str);
    putchar('\n');
    fflush(stdout);
}

int
NetHackRL::rl_nhgetch()
{
    DEBUG_API("rl_nhgetch" << std::endl);
    ScopedStack s(win_proc_calls, "nhgetch");
    int i = instance->getch_method();
    return i;
}

int
NetHackRL::rl_nh_poskey(int *x, int *y, int *mod)
{
    nhUse(x);
    nhUse(y);
    nhUse(mod);

    ScopedStack s(win_proc_calls, "nh_poskey");
    int action = rl_nhgetch();
    DEBUG_API("rl_nh_poskey: " << action << std::endl);
    return action;
    // Not calling nh_poskey, but no extra logic necessary here.
}

void
NetHackRL::rl_nhbell()
{
    DEBUG_API("rl_nhbell" << std::endl);
    ScopedStack s(win_proc_calls, "nhbell");
    return tty_nhbell();
}

int
NetHackRL::rl_doprev_message()
{
    DEBUG_API("rl_doprev_message" << std::endl);
    ScopedStack s(win_proc_calls, "doprev_message");
    int result = tty_doprev_message();
    return result;
}

char
NetHackRL::rl_yn_function(const char *question_, const char *choices,
                          CHAR_P def)
{
    DEBUG_API("rl_yn_function" << std::endl);
    ScopedStack s(win_proc_calls, "yn_function");
    in_yn_function = true;
    char result = tty_yn_function(question_, choices, def);
    in_yn_function = false;
    return result;
}

void
NetHackRL::rl_getlin(const char *prompt, char *line)
{
    DEBUG_API("rl_getlin" << std::endl);
    ScopedStack s(win_proc_calls, "getlin");
    in_getlin = true;
    tty_getlin(prompt, line);
    in_getlin = false;
}

int
NetHackRL::rl_get_ext_cmd()
{
    DEBUG_API("rl_get_ext_cmd" << std::endl);
    ScopedStack s(win_proc_calls, "get_ext_cmd");
    return tty_get_ext_cmd();
}

void
NetHackRL::rl_number_pad(int i)
{
    DEBUG_API("rl_number_pad" << std::endl);
    ScopedStack s(win_proc_calls, "number_pad");
    tty_number_pad(i);
}

void
NetHackRL::rl_delay_output()
{
    DEBUG_API("rl_delay_output" << std::endl);
    // No call to tty_delay_output() as we don't actually want delays.
}

void
NetHackRL::rl_start_screen()
{
    DEBUG_API("rl_start_screen" << std::endl);
    ScopedStack s(win_proc_calls, "start_screen");
    tty_start_screen();
}

void
NetHackRL::rl_end_screen()
{
    DEBUG_API("rl_end_screen" << std::endl);
    ScopedStack s(win_proc_calls, "end_screen");
    tty_end_screen();

    if (instance)
        // The only way instance can still be around is in an error situation.
        // Unfortunately, ZQM doesn't close properly when destructed via
        // global objects. So we do it here.
        instance.reset(nullptr);
}

void
NetHackRL::rl_outrip(winid wid, int how, time_t when)
{
    DEBUG_API("rl_outrip" << std::endl);
    genl_outrip(wid, how, when);
}

char *
NetHackRL::rl_getmsghistory(BOOLEAN_P init)
{
    DEBUG_API("rl_getmsghistory" << std::endl);
    return tty_getmsghistory(init);
}

void
NetHackRL::rl_putmsghistory(const char *msg, BOOLEAN_P is_restoring)
{
    DEBUG_API("rl_putmsghistory" << std::endl);
    tty_putmsghistory(msg, is_restoring);
}

void
NetHackRL::rl_status_init()
{
    DEBUG_API("rl_status_init" << std::endl);
    ScopedStack s(win_proc_calls, "status_init");
    tty_status_init();
}

void
NetHackRL::rl_status_update(int fldidx, genericptr_t ptr, int chg,
                            int percent, int color, unsigned long *colormasks)
{
    DEBUG_API("rl_status_update" << std::endl);

    ScopedStack s(win_proc_calls, "status_update");
    instance->status_update_method(fldidx, ptr, chg, percent, color,
                                   colormasks);
#ifdef STATUS_HILITES
    tty_status_update(fldidx, ptr, chg, percent, color, colormasks);
#endif
}

static void
rl_update_positionbar(char *chrs)
{
    DEBUG_API("rl_update_positionbar" << std::endl);
#ifdef POSITIONBAR
    tty_update_positionbar(chrs);
#endif
}

} // namespace nethack_rl

struct window_procs rl_procs = {
    "rl",
    (WC_COLOR | WC_HILITE_PET | WC_INVERSE | WC_EIGHT_BIT_IN
     | WC_PERM_INVENT),
    (0
#if defined(SELECTSAVED)
     | WC2_SELECTSAVED
#endif
#if defined(STATUS_HILITES)
     | WC2_HILITE_STATUS | WC2_HITPOINTBAR | WC2_FLUSH_STATUS
     | WC2_RESET_STATUS
#endif
     | WC2_DARKGRAY | WC2_SUPPRESS_HIST | WC2_STATUSLINES),
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1 }, /* color availability */
    nethack_rl::NetHackRL::rl_init_nhwindows,
    nethack_rl::NetHackRL::rl_player_selection,
    nethack_rl::NetHackRL::rl_askname,
    nethack_rl::NetHackRL::rl_get_nh_event,
    nethack_rl::NetHackRL::rl_exit_nhwindows,
    nethack_rl::NetHackRL::rl_suspend_nhwindows,
    nethack_rl::NetHackRL::rl_resume_nhwindows,
    nethack_rl::NetHackRL::rl_create_nhwindow,
    nethack_rl::NetHackRL::rl_clear_nhwindow,
    nethack_rl::NetHackRL::rl_display_nhwindow,
    nethack_rl::NetHackRL::rl_destroy_nhwindow,
    nethack_rl::NetHackRL::rl_curs,
    nethack_rl::NetHackRL::rl_putstr,
    genl_putmixed,
    nethack_rl::NetHackRL::rl_display_file,
    nethack_rl::NetHackRL::rl_start_menu,
    nethack_rl::NetHackRL::rl_add_menu,
    nethack_rl::NetHackRL::rl_end_menu,
    nethack_rl::NetHackRL::rl_select_menu,
    genl_message_menu, /* no need for X-specific handling */
    nethack_rl::NetHackRL::rl_update_inventory,
    nethack_rl::NetHackRL::rl_mark_synch,
    nethack_rl::NetHackRL::rl_wait_synch,
#ifdef CLIPPING
    nethack_rl::NetHackRL::rl_cliparound,
#endif
#ifdef POSITIONBAR
    nethack_rl::rl_update_positionbar,
#endif
    nethack_rl::NetHackRL::rl_print_glyph,
    // NetHackRL::rl_print_glyph_compose,
    nethack_rl::NetHackRL::rl_raw_print,
    nethack_rl::NetHackRL::rl_raw_print_bold,
    nethack_rl::NetHackRL::rl_nhgetch,
    nethack_rl::NetHackRL::rl_nh_poskey,
    nethack_rl::NetHackRL::rl_nhbell,
    nethack_rl::NetHackRL::rl_doprev_message,
    nethack_rl::NetHackRL::rl_yn_function,
    nethack_rl::NetHackRL::rl_getlin,
    nethack_rl::NetHackRL::rl_get_ext_cmd,
    nethack_rl::NetHackRL::rl_number_pad,
    nethack_rl::NetHackRL::rl_delay_output,
#ifdef CHANGE_COLOR /* only a Mac option currently */
    donull,
    donull,
    donull,
    donull,
#endif
    /* other defs that really should go away (they're tty specific) */
    nethack_rl::NetHackRL::rl_start_screen,
    nethack_rl::NetHackRL::rl_end_screen,
#ifdef GRAPHIC_TOMBSTONE
    nethack_rl::NetHackRL::rl_outrip,
#else
    genl_outrip,
#endif
    tty_preference_update,
    nethack_rl::NetHackRL::rl_getmsghistory,
    nethack_rl::NetHackRL::rl_putmsghistory,
    nethack_rl::NetHackRL::rl_status_init,
    genl_status_finish,
    tty_status_enablefield,
    nethack_rl::NetHackRL::rl_status_update,
    genl_can_suspend_yes,
};
