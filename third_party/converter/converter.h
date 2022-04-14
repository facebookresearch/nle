#ifndef CONVERTER_H
#define CONVERTER_H

#include <stdint.h>
#include <sys/time.h>

#define CONV_OK               1
#define CONV_STREAM_END       0
#define CONV_CRITICAL_ERROR   (-1)
#define CONV_HEADER_ERROR     (-2)
#define CONV_BODY_ERROR       (-3)
#define CONV_FILE_ERROR       (-4)

typedef struct Header {
  struct timeval tv;
  int len;
  char channel;
} Header;

typedef struct CharPtr {
  char *ptr;
  char *cur;
  char *end;
} CharPtr;

typedef struct SignedCharPtr {
  signed char *ptr;
  signed char *cur;
  signed char *end;
} SignedCharPtr;

typedef struct Int16Ptr {
  int16_t *ptr;
  int16_t *cur;
  int16_t *end;
} Int16Ptr;

typedef struct Int64Ptr {
  int64_t *ptr;
  int64_t *cur;
  int64_t *end;
} Int64Ptr;

typedef struct Conversion {
  void *vt; /* TMT object. */

  int is_v2; /* Are we reading a v2 ttyrec, or a v1 ttyrec? */

  size_t rows; /* Number of returned (cropped) rows. */
  size_t cols; /* Number of returned (cropped) columns. */

  size_t term_rows; /* Number of terminal (rendered) rows. */
  size_t term_cols; /* Number of terminal (rendered) columns. */

  CharPtr chars;       /* Array to fill chars in */
  SignedCharPtr colors; /* Array to fill colors in */
  Int16Ptr cursors;    /* Array to fill current cursor positions in */
  Int64Ptr timestamps; /* Array to fill timestamp values in */
  CharPtr inputs; /* Array to fill inputs values in */

  size_t remaining; /* Remaining (free) number of frames in buffers */

  Header header; /* Most recently read header. */

  void *bfp; /* Pointer to current ttyrec BZFILE. */
  char *buf; /* Buffer for read data. */
} Conversion;

Conversion *conversion_create(size_t rows, size_t cols, size_t term_rows,
                              size_t term_cols, int is_v2);
void conversion_set_buffers(Conversion *c, char *chars, size_t chars_size,
                            signed char *colors, size_t colors_size,
                            int16_t *curcurs, size_t curcurs_size,
                            int64_t *timestamps, size_t timestamps_size,
                            char *inputs, size_t inputs_size);
int conversion_load_ttyrec(Conversion *c, FILE *f);
int conversion_convert_frames(Conversion *c);
int conversion_close(Conversion *c);

#endif /* CONVERTER_H */
