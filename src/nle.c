
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#include <tmt.h>

#define NEED_VARARGS
#ifdef MONITOR_HEAP
#undef MONITOR_HEAP
#endif
#include "hack.h"

#include "dlb.h"

#include "nle.h"

#ifdef NLE_BZ2_TTYRECS
#include <bzlib.h>
#endif

#define STACK_SIZE (1 << 15) // 32KiB

#ifndef __has_feature
#define __has_feature(x) 0 // Compatibility with non-clang compilers.
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#endif

extern int unixmain(int, char **);

signed char
vt_char_color_extract(TMTCHAR *c)
{
    /* We pick out the colors in the enum tmt_color_t. These match the order
     * found standard in IBM color graphics, and are the same order as those
     * found in src/color.h. We take the values from color.h, and choose
     * default to be bright black (NO_COLOR) as nethack does.
     *
     * Finally we indicate whether the color is reverse, by indicating the
     * sign
     * of the final integer.
     */
    signed char color = 0;
    switch (c->a.fg) {
    case (TMT_COLOR_DEFAULT):
        color =
            (c->c == 32) ? CLR_BLACK : CLR_GRAY; // ' ' is BLACK else WHITE
        break;
    case (TMT_COLOR_BLACK):
        color = (c->a.bold) ? NO_COLOR : CLR_BLACK; // c = 8:0
        break;
    case (TMT_COLOR_RED):
        color = (c->a.bold) ? CLR_ORANGE : CLR_RED; // c = 9:1
        break;
    case (TMT_COLOR_GREEN):
        color = (c->a.bold) ? CLR_BRIGHT_GREEN : CLR_GREEN; // c = 10:2
        break;
    case (TMT_COLOR_YELLOW):
        color = (c->a.bold) ? CLR_YELLOW : CLR_BROWN; // c = 11:3
        break;
    case (TMT_COLOR_BLUE):
        color = (c->a.bold) ? CLR_BRIGHT_BLUE : CLR_BLUE; // c = 12:4
        break;
    case (TMT_COLOR_MAGENTA):
        color = (c->a.bold) ? CLR_BRIGHT_MAGENTA : CLR_MAGENTA; // c = 13:5
        break;
    case (TMT_COLOR_CYAN):
        color = (c->a.bold) ? CLR_BRIGHT_CYAN : CLR_CYAN; // c = 14:6
        break;
    case (TMT_COLOR_WHITE):
        color = (c->a.bold) ? CLR_WHITE : CLR_GRAY; // c = 15:7
        break;
    }

    if (c->a.reverse) {
        color += CLR_MAX;
    }
    return color;
}

void
nle_vt_callback(tmt_msg_t m, TMT *vt, const void *a, void *p)
{
    const TMTSCREEN *s = tmt_screen(vt);
    const TMTPOINT *cur = tmt_cursor(vt);

    nle_ctx_t *nle = (nle_ctx_t *) p;
    if (!nle || !nle->observation) {
        return;
    }

    switch (m) {
    case TMT_MSG_BELL:
        break;

    case TMT_MSG_UPDATE:
        for (size_t r = 0; r < s->nline; r++) {
            if (s->lines[r]->dirty) {
                for (size_t c = 0; c < s->ncol; c++) {
                    size_t offset = (r * NLE_TERM_CO) + c;
                    TMTCHAR *tmt_c = &(s->lines[r]->chars[c]);

                    if (nle->observation->tty_chars) {
                        nle->observation->tty_chars[offset] = tmt_c->c;
                    }

                    if (nle->observation->tty_colors) {
                        nle->observation->tty_colors[offset] =
                            vt_char_color_extract(tmt_c);
                    }
                }
            }
        }
        tmt_clean(vt);
        break;

    case TMT_MSG_ANSWER:
        break;

    case TMT_MSG_MOVED:
        if (nle->observation->tty_cursor) {
            // cast from size_t is safe from overflow, since r,c < 256
            nle->observation->tty_cursor[0] = (unsigned char) cur->r;
            nle->observation->tty_cursor[1] = (unsigned char) cur->c;
        }
        break;

    case TMT_MSG_CURSOR:
        break;
    }
}

nle_ctx_t *
init_nle(FILE *ttyrec, nle_obs *obs)
{
    nle_ctx_t *nle = malloc(sizeof(nle_ctx_t));

    assert(ttyrec != NULL);
    nle->ttyrec = ttyrec;

#ifdef NLE_BZ2_TTYRECS
    int bzerror;
    nle->ttyrec_bz2 = BZ2_bzWriteOpen(&bzerror, ttyrec, 9, 0, 0);
    assert(bzerror == BZ_OK);
#endif

    nle->observation = obs;

    TMT *vterminal = tmt_open(LI, CO, nle_vt_callback, nle, NULL);
    assert(!vterminal);
    nle->vterminal = vterminal;

    nle->outbuf_write_ptr = nle->outbuf;
    nle->outbuf_write_end = nle->outbuf + sizeof(nle->outbuf);

    return nle;
}

/* TODO: Consider copying the relevant parts of main() in unixmain.c. */
void
mainloop(fcontext_transfer_t ctx_transfer)
{
    current_nle_ctx->returncontext = ctx_transfer.ctx;
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
    /* ASan isn't happy with fcontext's assembly.
     * See: https://bugs.llvm.org/show_bug.cgi?id=27627 and
     * https://github.com/boostorg/coroutine/issues/30#issuecomment-325578344
     * TODO: I don't understand why __sanitizer_(start/finish)_switch_fiber
     * doesn't work here.
     */
    fcontext_stack_t *stack = &current_nle_ctx->stack;
    ASAN_UNPOISON_MEMORY_REGION((char *) stack->sptr - stack->ssize,
                                stack->ssize);
#endif

    char *argv[1] = { "nethack" };

    unixmain(1, argv);
}

boolean
write_data(void *buf, int length)
{
    nle_ctx_t *nle = current_nle_ctx;
#ifdef NLE_BZ2_TTYRECS
    int bzerror;
    BZ2_bzWrite(&bzerror, nle->ttyrec_bz2, buf, length);
    assert(bzerror == BZ_OK);
#else
    assert(fwrite(buf, 1, length, nle->ttyrec) == length);
#endif
    return TRUE;
}

boolean
write_header(int length, unsigned char channel)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    int buffer[3];
    buffer[0] = tv.tv_sec;
    buffer[1] = tv.tv_usec;
    buffer[2] = length;

    /* Assumes little endianness */
    write_data(buffer, 3 * sizeof(int));
    write_data(&channel, 1);

    return TRUE;
}

/* win/tty only calls fflush(stdout). */
int
nle_fflush(FILE *stream)
{
    /* Only act on fflush(stdout). */
    if (stream != stdout) {
        fprintf(stderr,
                "Warning: nle_flush called with unexpected FILE pointer %d ",
                (int) stream);
        return fflush(stream);
    }
    nle_ctx_t *nle = current_nle_ctx;

    ssize_t length = nle->outbuf_write_ptr - nle->outbuf;
    if (length == 0)
        return 0;

    write_header(length, 0);
    write_data(nle->outbuf, length);

    nle_obs *obs = nle->observation;
    if (obs->tty_chars || obs->tty_colors || obs->tty_cursor) {
        tmt_write(nle->vterminal, nle->outbuf, length);
    }
    nle->outbuf_write_ptr = nle->outbuf;

#ifdef NLE_BZ2_TTYRECS
    return 0;
#else
    return fflush(nle->ttyrec);
#endif
}

/*
 * NetHack prints most of its output via putchar. We do our
 * own buffering.
 */
int
nle_putchar(int c)
{
    nle_ctx_t *nle = current_nle_ctx;
    if (nle->outbuf_write_ptr >= nle->outbuf_write_end) {
        nle_fflush(stdout);
    }
    *nle->outbuf_write_ptr++ = c;
    return c;
}

/*
 * Used in place of xputs from termcap.c. Not using
 * the tputs padding logic from tclib.c.
 */
void
nle_xputs(const char *str)
{
    int c;
    const char *p = str;

    if (!p || !*p)
        return;

    while ((c = *p++) != '\0') {
        nle_putchar(c);
    }
}

/*
 * puts seems to be called only by tty_raw_print and tty_raw_print_bold.
 * We could probably override this in winrl instead.
 */
int
nle_puts(const char *str)
{
    if (!*str) /* At exit, an empty string gets printed in tty_raw_print. */
        return 0;

    int val = fputs(str, stdout);
    putc('\n', stdout); /* puts includes a newline, fputs doesn't */
    return val;
}

/* Necessary for initial observation struct. */
nle_obs *
nle_get_obs()
{
    return current_nle_ctx->observation;
}

void *
nle_yield(void *notdone)
{
    nle_fflush(stdout);
    fcontext_transfer_t t =
        jump_fcontext(current_nle_ctx->returncontext, notdone);
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
    fcontext_stack_t *stack = &current_nle_ctx->stack;
    ASAN_UNPOISON_MEMORY_REGION((char *) stack->sptr - stack->ssize,
                                stack->ssize);
#endif

    if (notdone)
        current_nle_ctx->returncontext = t.ctx;

    return t.data;
}

void
nethack_exit(int status)
{
    if (status) {
        fprintf(stderr, "NetHack exit with status %i\n", status);
    }
    nle_yield(NULL);
}

nle_seeds_init_t *nle_seeds_init;

/* See rng.c. */
extern int FDECL(whichrng, (int FDECL((*fn), (int) )));

/* See hacklib.c. */
extern int FDECL(set_random, (unsigned long, int FDECL((*fn), (int) )));
/* An appropriate version of this must always be provided in
   port-specific code somewhere. It returns a number suitable
   as seed for the random number generator */
extern unsigned long NDECL(sys_random_seed);

/*
 * Initializes the random number generator.
 * Originally in hacklib.c.
 */
void
init_random(int FDECL((*fn), (int) ))
{
    if (!nle_seeds_init) {
        set_random(sys_random_seed(), fn);
        return;
    }
    set_random(nle_seeds_init->seeds[whichrng(fn)], fn);
    has_strong_rngseed = nle_seeds_init->reseed;
}

nle_ctx_t *
nle_start(nle_obs *obs, FILE *ttyrec, nle_seeds_init_t *seed_init)
{
    /* Set CO and LI to control ttyrec output size. */
    CO = NLE_TERM_CO;
    LI = NLE_TERM_LI;

    nle_ctx_t *nle = init_nle(ttyrec, obs);
    nle_seeds_init = seed_init;

    nle->stack = create_fcontext_stack(STACK_SIZE);
    nle->generatorcontext =
        make_fcontext(nle->stack.sptr, nle->stack.ssize, mainloop);

    current_nle_ctx = nle;
    fcontext_transfer_t t = jump_fcontext(nle->generatorcontext, NULL);
    nle->generatorcontext = t.ctx;
    nle->done = (t.data == NULL);
    obs->done = nle->done;
    nle_seeds_init =
        NULL; /* Don't set to *these* seeds on subsequent reseeds, if any. */

    return nle;
}

nle_ctx_t *
nle_step(nle_ctx_t *nle, nle_obs *obs)
{
    current_nle_ctx = nle;
    nle->observation = obs;
    write_header(1, 1);
    write_data(&obs->action, 1);
    fcontext_transfer_t t = jump_fcontext(nle->generatorcontext, obs);
    nle->generatorcontext = t.ctx;
    nle->done = (t.data == NULL);
    obs->done = nle->done;

    return nle;
}

void
nle_end(nle_ctx_t *nle)
{
    if (!nle->done) {
        /* Reset without closing nethack. Need free memory, etc.
         * this is what nh_terminate in end.c does. I hope it's enough. */
        if (!program_state.panicking) {
            freedynamicdata();
            dlb_cleanup();
        }
    }
    nle_fflush(stdout);

#ifdef NLE_BZ2_TTYRECS
    int bzerror;
    BZ2_bzWriteClose(&bzerror, nle->ttyrec_bz2, 0, NULL, NULL);
    assert(bzerror == BZ_OK);
#endif

    tmt_close(nle->vterminal);

    destroy_fcontext_stack(&nle->stack);
    free(nle);
}

void
nle_set_seed(nle_ctx_t *nle, unsigned long core, unsigned long disp,
             boolean reseed)
{
    /* Keep up to date with rnglist[] in rnd.c. */
    set_random(core, rn2);
    set_random(disp, rn2_on_display_rng);

    /* Determines logic in reseed_random() in hacklib.c. */
    has_strong_rngseed = reseed;
};

extern unsigned long nle_seeds[];

void
nle_get_seed(nle_ctx_t *nle, unsigned long *core, unsigned long *disp,
             boolean *reseed)
{
    *core = nle_seeds[0];
    *disp = nle_seeds[1];
    *reseed = has_strong_rngseed;
}

/* From unixtty.c */
/* fatal error */
/*VARARGS1*/
void error
VA_DECL(const char *, s)
{
    VA_START(s);
    VA_INIT(s, const char *);

    if (iflags.window_inited)
        exit_nhwindows((char *) 0); /* for tty, will call settty() */

    fprintf(stderr, s, VA_ARGS);
    fprintf(stderr, "\n");
    VA_END();
    nethack_exit(EXIT_FAILURE);
}

/* From unixtty.c */
char erase_char, intr_char, kill_char;

void
gettty()
{
    /* Should set erase_char, intr_char, kill_char */
}

void
settty(const char *s)
{
    end_screen();
    if (s)
        raw_print(s);
}

void
setftty()
{
    start_screen();

    iflags.cbreak = ON;
    iflags.echo = OFF;
}

void
intron()
{
}

void
introff()
{
}

#ifdef __linux__ /* via Jesse Thilo and Ben Gertzfield */
#include <sys/ioctl.h>
#include <sys/vt.h>

int linux_flag_console = 0;

void NDECL(linux_mapon);
void NDECL(linux_mapoff);
void NDECL(check_linux_console);
void NDECL(init_linux_cons);

void
linux_mapon()
{
#ifdef TTY_GRAPHICS
    if (WINDOWPORT("tty") && linux_flag_console) {
        write(1, "\033(B", 3);
    }
#endif
}

void
linux_mapoff()
{
#ifdef TTY_GRAPHICS
    if (WINDOWPORT("tty") && linux_flag_console) {
        write(1, "\033(U", 3);
    }
#endif
}

void
check_linux_console()
{
    struct vt_mode vtm;

    if (isatty(0) && ioctl(0, VT_GETMODE, &vtm) >= 0) {
        linux_flag_console = 1;
    }
}

#ifdef NEED_HAS_COLORS_DECL
int has_colors();
#endif

void
init_linux_cons()
{
#ifdef TTY_GRAPHICS
    if (WINDOWPORT("tty") && linux_flag_console) {
        atexit(linux_mapon);
        linux_mapoff();
#ifdef TEXTCOLOR
        if (has_colors())
            iflags.use_color = TRUE;
#endif
    }
#endif
}
#endif /* __linux__ */
