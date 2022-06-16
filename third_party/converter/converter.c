/*
 *  ttyrec to array converter.
 *  TODOs:
 *    * Terminal size?
 *    * Handle DECgraphics and IBMgraphics
 *    * Output attributes: reverse, fg color, bg color
 *      (plus bold, dim, underline, blink, invisible ...)
 */

#ifdef NDEBUG
/* Enable assert */
#undef NDEBUG
#endif

#include <assert.h>
#include <bzlib.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "stripgfx.h"
#include "tmt.h"

#include "converter.h"

#define UNUSED(x) (void)(x)

/* Callback for libtmt. */
void callback(tmt_msg_t, TMT *, const void *, void *);


/*
 * The color scheme used is tailored for an IBM PC.  It consists of the
 * standard 8 colors, followed by their bright counterparts.  There are
 * exceptions, these are listed below.	Bright black doesn't mean very
 * much, so it is used as the "default" foreground color of the screen.
 */
#define CLR_BLACK 0
#define CLR_RED 1
#define CLR_GREEN 2
#define CLR_BROWN 3 /* on IBM, low-intensity yellow is brown */
#define CLR_BLUE 4
#define CLR_MAGENTA 5
#define CLR_CYAN 6
#define CLR_GRAY 7 /* low-intensity white */
#define NO_COLOR 8
#define CLR_ORANGE 9
#define CLR_BRIGHT_GREEN 10
#define CLR_YELLOW 11
#define CLR_BRIGHT_BLUE 12
#define CLR_BRIGHT_MAGENTA 13
#define CLR_BRIGHT_CYAN 14
#define CLR_WHITE 15
#define CLR_MAX 16

/* The "half-way" point for tty based color systems.  This is used in */
/* the tty color setup code.  (IMHO, it should be removed - dean).    */
#define BRIGHT 8


/* Taken from nle.c -- github.com/facebookresearch/nle */
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
    case (TMT_COLOR_MAX):
        perror("Invalid color encountered.");
        assert(false);
        break;
    case (TMT_COLOR_DEFAULT):
    default:  // Bad IBM Graphic can lead to case: (0), so we add default
        color =
            (c->c == 32) ? CLR_BLACK : CLR_GRAY; // ' ' is BLACK else WHITE
        break;
    }

    if (c->a.reverse) {
        color += CLR_MAX;
    }
    return color;
}


int read_header(BZFILE *bfp, Header *h, size_t version) {
  int buf[3];
  int bzerror;
  BZ2_bzRead(&bzerror, bfp, buf, sizeof(int) * 3);
  if (bzerror != BZ_OK) {
    /* This could be BZ_STREAM_END, the logical end of a stream.
       We still stop in that case. */
    if (bzerror == BZ_STREAM_END) return CONV_STREAM_END;
    return CONV_HEADER_ERROR;
  }

  h->tv.tv_sec = buf[0];
  h->tv.tv_usec = buf[1];
  h->len = buf[2];
  h->channel = 0;

  if (version > 1) {
    /* NLE-based ttyrecs read have single-byte "channel" which codifies what 
    kind of information one is in the buffer. Here we read into the channel. */
    BZ2_bzRead(&bzerror, bfp, &h->channel, 1);
    if (bzerror != BZ_OK) {
      if (bzerror == BZ_STREAM_END) return CONV_STREAM_END;
      return CONV_HEADER_ERROR;
    }
  }
  
  if (h->len == 0) {
    /* Some ttyrecs seem to result in all 0 header. In this case
     * treat as non-fatal error, drop the ttyrec and continue as if end of stream.
     * eg: /datasets01/altorg/111720/GrieferBonez4/2014-08-27.00:42:39.ttyrec.bz2 */
    fprintf(stderr, "Ttyrec has zero-length header\n");
    return CONV_FILE_ERROR;
  }

  return CONV_OK;
}

int ttyread(BZFILE *bfp, Header *h, char **buf, size_t version) {
  int status = read_header(bfp, h, version);
  if (status != CONV_OK) {
    return status;
  }

  *buf = realloc(*buf, h->len);
  if (*buf == NULL) {
    perror("malloc");
    return CONV_CRITICAL_ERROR;
  }

  int bzerror;
  int length = BZ2_bzRead(&bzerror, bfp, *buf, h->len);
  if (bzerror != BZ_OK || length != h->len) {
    if (bzerror == BZ_STREAM_END) return CONV_STREAM_END;
    fprintf(stderr, "bzRead failed with return code %d (read %d bytes)\n",
            bzerror, length);
    return CONV_BODY_ERROR;
  }
  return CONV_OK;
}

Conversion *conversion_create(size_t rows, size_t cols, size_t term_rows,
                              size_t term_cols, size_t version) {
  static bool stripgfx_init = false;
  if (!stripgfx_init) {
    populate_gfx_arrays();
    stripgfx_init = true;
  }

  Conversion *c = malloc(sizeof(Conversion));
  if (!c) return NULL;
  c->version = version;
  c->rows = rows;
  c->cols = cols;
  if (!term_rows) term_rows = rows;
  if (!term_cols) term_cols = cols;
  assert(rows <= term_rows && cols <= term_cols);
  c->chars = (UnsignedCharPtr){0};
  c->colors = (SignedCharPtr){0};
  c->cursors = (Int16Ptr){0};
  c->timestamps = (Int64Ptr){0};
  c->inputs = (UnsignedCharPtr){0};
  c->scores = (Int32Ptr){0};
  c->remaining = 0;
  c->buf = NULL;
  bool wrap = (version != 1);
  if (!wrap) {
    /* For old ttyrecs where we don't wrap, we make cols one character wider.
    This last character will keep getting overwritten. This last column is
    not copied to our buffers.*/
    term_cols += 1;
  }
  c->vt = tmt_open(term_rows, term_cols, callback, c, NULL, wrap);
  if (!c->vt) {
    perror("could not allocate terminal");
    free(c);
    return NULL;
  }
  c->bfp = NULL;
  return c;
}

void conversion_set_buffers(Conversion *c, unsigned char *chars, size_t chars_size,
                            signed char * colors, size_t colors_size,
                            int16_t *cursors, size_t cursors_size,
                            int64_t *timestamps, size_t timestamps_size,
                            unsigned char *inputs, size_t inputs_size,
                            int32_t *scores, size_t scores_size) {
  assert(chars_size % (c->rows * c->cols) == 0);
  c->remaining = chars_size / (c->rows * c->cols);

  assert(cursors_size % 2 == 0);
  assert(cursors_size / 2 == c->remaining);

  assert(timestamps_size == c->remaining);

  c->chars = (UnsignedCharPtr){chars, chars, chars + chars_size};
  c->colors = (SignedCharPtr){colors, colors, colors + colors_size};
  c->cursors = (Int16Ptr){cursors, cursors, cursors + cursors_size};
  c->timestamps =
      (Int64Ptr){timestamps, timestamps, timestamps + timestamps_size};
  c->inputs =
      (UnsignedCharPtr){inputs, inputs, inputs + inputs_size};
  c->scores =
      (Int32Ptr){scores, scores, scores + scores_size};
}

int conversion_load_ttyrec(Conversion *c, FILE *f) {
  int bzerror;
  if (c->bfp) {
    BZ2_bzReadClose(&bzerror, c->bfp);
  }

  c->bfp = BZ2_bzReadOpen(&bzerror, f, 0, 1, NULL, 0);
  if (bzerror != BZ_OK) {
    perror("Could not open bzip2 file");
    BZ2_bzReadClose(&bzerror, c->bfp);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void write_to_buffers(Conversion *conv);

/* Returns 1 at end of buffer, 0 at end of input, -1 on failure. */
int conversion_convert_frames(Conversion *c) {
  if (!c->bfp || !c->chars.cur) return CONV_CRITICAL_ERROR;

  int status = CONV_OK;

  while (c->remaining) {
    status = ttyread(c->bfp, &c->header, &c->buf, c->version);
    if (status != CONV_OK) break;

    if (c->version > 1){
      /* NLE-based ttyrecs have a channel which codifies what type of
       * information we are encoding. 
       * 
       * V1: Order: 0 - No "channel", write only to update terminal
       * V2: [0 1 0 1 ...]
       *     Channel 0 -> update terminal/state
       *     Channel 1 -> we have an action: write state + action to buffers
       * V3: [0 2 1 0 2 1 ...]
       *     Channel 0 -> update terminal/state
       *     Channel 2 -> we have an reward: write reward only
       *     Channel 1 -> we have an action: write state + action to buffers 
       * NB. Will only end up writing when an action is given. */
      if (c->header.channel == 0) {
        tmt_write(c->vt, c->buf, c->header.len);
      } else {
        write_to_buffers(c);
      }
    } else if (c->version == 1) {
      /* V1: We write every frame to buffer (unclear when actions taken) */
      tmt_write(c->vt, c->buf, c->header.len);
      write_to_buffers(c);
    } else {
      perror("Unrecognized ttyrec version");
    }
  }

  return status;
}

void write_to_buffers(Conversion *conv) {
  if (conv->version > 1)  {
    if (conv->header.channel == 2) {
      /* V3: Write just the reward. Do not write the screen. */
      memcpy(conv->scores.cur++, conv->buf, sizeof(*conv->scores.cur));
      return;
    }
    if (conv->header.channel == 1) {
      /* V2: Write the action, then continue to flush the screen too. */
      *conv->inputs.cur++ = conv->buf[0];
    }
  }
  
  const TMTSCREEN *scr = tmt_screen(conv->vt);
  for (size_t r = 0; r < conv->rows; ++r) {
    for (size_t c = 0; c < conv->cols; ++c) {
      assert(conv->chars.cur < conv->chars.end);
      assert(scr->lines[r]->chars[c].c < 256);
      *conv->chars.cur++ = strip_gfx(scr->lines[r]->chars[c].c, scr->lines[r]->chars[c].a.dec);
      *conv->colors.cur++ = vt_char_color_extract(&(scr->lines[r]->chars[c]));
    }
  }

  const TMTPOINT *cur = tmt_cursor(conv->vt);
  *conv->cursors.cur++ = cur->r;
  *conv->cursors.cur++ = cur->c;

  int64_t usec = 1000000 * (int64_t)conv->header.tv.tv_sec;
  *conv->timestamps.cur++ = usec + (int64_t)conv->header.tv.tv_usec;

  --conv->remaining;

}

int conversion_close(Conversion *c) {
  if (!c || !c->vt) {
    perror("Conversion not loaded");
    return 1;
  }
  tmt_close(c->vt);
  if (c->bfp) {
    int bzerror;
    BZ2_bzReadClose(&bzerror, c->bfp);
  }
  if (c->buf) free(c->buf);
  free(c);
  return EXIT_SUCCESS;
}

void callback(tmt_msg_t m, TMT *vt, const void *a, void *p) {
  UNUSED(m);
  UNUSED(a);
  UNUSED(p);

  tmt_clean(vt);
}
