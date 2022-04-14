#ifndef CONVERTER_H
#define CONVERTER_H

#include <stdint.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C"{
#endif 

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

typedef struct UnsignedCharPtr {
  unsigned char *ptr;
  unsigned char *cur;
  unsigned char *end;
} UnsignedCharPtr;

typedef struct SignedCharPtr {
  signed char *ptr;
  signed char *cur;
  signed char *end;
} SignedCharPtr;

typedef struct UInt16Ptr {
  uint16_t *ptr;
  uint16_t *cur;
  uint16_t *end;
} UInt16Ptr;

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

  UnsignedCharPtr chars;       /* Array to fill chars in */
  SignedCharPtr colors; /* Array to fill colors in */
  UInt16Ptr cursors;    /* Array to fill current cursor positions in */
  Int64Ptr timestamps; /* Array to fill timestamp values in */
  UnsignedCharPtr inputs; /* Array to fill inputs values in */

  size_t remaining; /* Remaining (free) number of frames in buffers */

  Header header; /* Most recently read header. */

  void *bfp; /* Pointer to current ttyrec BZFILE. */
  char *buf; /* Buffer for read data. */
} Conversion;

Conversion *conversion_create(size_t rows, size_t cols, size_t term_rows,
                              size_t term_cols, int is_v2);
void conversion_set_buffers(Conversion *c, unsigned char *chars, size_t chars_size,
                            signed char *colors, size_t colors_size,
                            uint16_t *curcurs, size_t curcurs_size,
                            int64_t *timestamps, size_t timestamps_size,
                            unsigned char *inputs, size_t inputs_size);
int conversion_load_ttyrec(Conversion *c, FILE *f);
int conversion_convert_frames(Conversion *c);
int conversion_close(Conversion *c);

#ifdef __cplusplus
}
#endif

#endif /* CONVERTER_H */
