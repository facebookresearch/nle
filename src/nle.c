
#include <assert.h>
#include <sys/time.h>

#include <signal.h>
#include <string.h>

#define NEED_VARARGS
#include "hack.h"

#include "dlb.h"

#include "nle.h"

#define STACK_SIZE (1 << 22) // 4MB

/* We are fine with whatever. */
boolean
authorize_wizard_mode()
{
    return TRUE;
}

boolean check_user_string(optstr) char *optstr;
{
    return TRUE;
}

void port_insert_pastebuf(buf) char *buf;
{
}

/* Copied from unixmain.c. */
unsigned long
sys_random_seed()
{
    unsigned long seed = 0L;
    unsigned long pid = (unsigned long) getpid();
    boolean no_seed = TRUE;
#ifdef DEV_RANDOM
    FILE *fptr;

    fptr = fopen(DEV_RANDOM, "r");
    if (fptr) {
        fread(&seed, sizeof(long), 1, fptr);
        has_strong_rngseed = TRUE; /* decl.c */
        no_seed = FALSE;
        (void) fclose(fptr);
    } else {
        /* leaves clue, doesn't exit */
        paniclog("sys_random_seed", "falling back to weak seed");
    }
#endif
    if (no_seed) {
        seed = (unsigned long) getnow(); /* time((TIME_type) 0) */
        /* Quick dirty band-aid to prevent PRNG prediction */
        if (pid) {
            if (!(pid & 3L))
                pid -= 1L;
            seed *= pid;
        }
    }
    return seed;
}

/* Copied from unixmain.c. */
void sethanguphandler(handler) void FDECL((*handler), (int) );
{
#ifdef SA_RESTART
    /* don't want reads to restart.  If SA_RESTART is defined, we know
     * sigaction exists and can be used to ensure reads won't restart.
     * If it's not defined, assume reads do not restart.  If reads restart
     * and a signal occurs, the game won't do anything until the read
     * succeeds (or the stream returns EOF, which might not happen if
     * reading from, say, a window manager). */
    struct sigaction sact;

    (void) memset((genericptr_t) &sact, 0, sizeof sact);
    sact.sa_handler = (SIG_RET_TYPE) handler;
    (void) sigaction(SIGHUP, &sact, (struct sigaction *) 0);
#ifdef SIGXCPU
    (void) sigaction(SIGXCPU, &sact, (struct sigaction *) 0);
#endif
#else /* !SA_RESTART */
    (void) signal(SIGHUP, (SIG_RET_TYPE) handler);
#ifdef SIGXCPU
    (void) signal(SIGXCPU, (SIG_RET_TYPE) handler);
#endif
#endif /* ?SA_RESTART */
}

nle_ctx_t *init_nle(outfile) FILE *outfile;
{
    nle_ctx_t *nle = malloc(sizeof(nle_ctx_t));

    if (!outfile) {
        outfile = fopen("/dev/null", "w");
    }
    nle->ttyrec = outfile;

    nle->outbuf_write_ptr = nle->outbuf;
    nle->outbuf_write_end = nle->outbuf + sizeof(nle->outbuf);

    return nle;
}

void
mainloop(fcontext_transfer_t ctx_transfer)
{
    current_nle_ctx->returncontext = ctx_transfer.ctx;

    early_init();

    g.hname = "nethack";
    g.hackpid = getpid();

    choose_windows("rl");

    /* TODO: Make nethack not require HACKDIR to be pwd. */
    const char *dir = HACKDIR;
    if (dir && chdir(dir) < 0) {
        perror(dir);
        error("Cannot chdir to %s.", dir);
    }

    strncpy(g.plname, "Agent", sizeof g.plname - 1);

#ifdef _M_UNIX
    check_sco_console();
#endif
#ifdef __linux__
    check_linux_console();
#endif
    initoptions();

    u.uhp = 1; /* prevent RIP on early quits */
    g.program_state.preserve_locks = 1;

    init_nhwindows(0, NULL); /* now we can set up window system */

#ifndef NO_SIGNAL
    sethanguphandler((SIG_RET_TYPE) hangup);
#endif

#ifdef _M_UNIX
    init_sco_cons();
#endif
#ifdef __linux__
    init_linux_cons();
#endif

    set_playmode(); /* sets plname to "wizard" for wizard mode */
    /* hide any hyphens from plnamesuffix() */
    g.plnamelen = (int) strlen(g.plname);

    /* strip role,race,&c suffix; calls askname() if plname[] is empty
       or holds a generic user name like "player" or "games" */
    plnamesuffix();

    dlb_init(); /* must be before newgame() */

    /*
     * Initialize the vision system.  This must be before mklev() on a
     * new game or before a level restore on a saved game.
     */
    vision_init();

    /* Creates and displays the game windows. */
    display_gamewindows();

    boolean resuming = FALSE;

    if (*g.plname) {
        /* TODO(heiner): Remove locks entirely.
           By default, this also checks that we're on a pty... */
        getlock();
        g.program_state.preserve_locks = 0; /* after getlock() */
    }

    if (restore_saved_game() != 0) {
        pline("Not restoring save file...");
        if (yn("Do you want to keep the save file?") == 'n') {
            (void) delete_savefile();
        }
    }

    if (!resuming) {
        player_selection();
        newgame();
    }

    moveloop(resuming);
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

    nle_ctx_t *nle = current_nle_ctx;

    /* Assumes little endianness */
    if (fwrite(buffer, sizeof(int), 3, nle->ttyrec) == 0) {
        assert(FALSE);
        return FALSE;
    }

    if (fputc((int) channel, nle->ttyrec) != (int) channel) {
        assert(FALSE);
        return FALSE;
    }

    return TRUE;
}

/* win/tty only calls fflush(stdout). */
int nle_fflush(stream) FILE *stream;
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
    /* TODO(heiner): Given that we do our own buffering, consider
     * using file descriptors instead of the ttyrec FILE*. */
    write_header(length, 0);
    fwrite(nle->outbuf, 1, length, nle->ttyrec);
    nle->outbuf_write_ptr = nle->outbuf;
    return fflush(nle->ttyrec);
}

/*
 * NetHack prints most of its output via putchar. We do our
 * own buffering.
 */
int nle_putchar(c) int c;
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
void nle_xputs(str) const char *str;
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
int nle_puts(str) const char *str;
{
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

    if (notdone)
        current_nle_ctx->returncontext = t.ctx;

    return t.data;
}

void nethack_exit(status) int status;
{
    nle_yield(NULL);
}

nle_ctx_t *nle_start(outfile, obs) FILE *outfile;
nle_obs *obs;
{
    nle_ctx_t *nle = init_nle(outfile);
    nle->observation = obs;

    nle->stack = create_fcontext_stack(STACK_SIZE);
    nle->generatorcontext =
        make_fcontext(nle->stack.sptr, nle->stack.ssize, mainloop);

    current_nle_ctx = nle;
    fcontext_transfer_t t = jump_fcontext(nle->generatorcontext, NULL);
    nle->generatorcontext = t.ctx;
    obs->done = (t.data == NULL);

    return nle;
}

nle_ctx_t *
nle_step(nle_ctx_t *nle, nle_obs *obs)
{
    current_nle_ctx = nle;
    nle->observation = obs;
    fcontext_transfer_t t = jump_fcontext(nle->generatorcontext, obs);
    nle->generatorcontext = t.ctx;
    obs->done = (t.data == NULL);

    return nle;
}

void
nle_end(nle_ctx_t *nle)
{
    destroy_fcontext_stack(&nle->stack);
    free(nle);
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

    Vprintf(s, VA_ARGS);
    (void) putchar('\n');
    VA_END();
    exit(EXIT_FAILURE);
}

/* From unixtty.c */
char erase_char, intr_char, kill_char;

void
gettty()
{
    /* Should set erase_char, intr_char, kill_char */
}

void settty(s) const char *s;
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
