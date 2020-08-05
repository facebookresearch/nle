
#include <assert.h>
#include <sys/time.h>

#include <string.h>

#define NEED_VARARGS
#include "hack.h"

#include "dlb.h"

#include "nle.h"

#define STACK_SIZE (1 << 15) // 32KiB

extern int unixmain(int, char **);

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

/* TODO: Consider copying the relevant parts of main() in unixmain.c. */
void
mainloop(fcontext_transfer_t ctx_transfer)
{
    current_nle_ctx->returncontext = ctx_transfer.ctx;

    char *argv[1] = { "nethack" };

    unixmain(1, argv);
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
    if (status) {
        fprintf(stderr, "NetHack exit with status %i\n", status);
    }
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

/* TODO: This doesn't properly free NetHack's memory. Fix. */
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
    nethack_exit(EXIT_FAILURE);
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
