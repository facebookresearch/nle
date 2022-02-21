#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "converter.h"

#define LEN 2
#define ROWS 24
#define COLS 80

void printtime(int64_t timestamp) {
  char tmbuf[64];
  time_t sec = timestamp / 1000000;
  int usec = timestamp % 1000000;

  struct tm* nowtm = localtime(&sec);
  strftime(tmbuf, sizeof tmbuf, "%Y-%m-%d %H:%M:%S", nowtm);
  printf("%s.%i\n", tmbuf, usec);
}

int main(void) {
  Conversion* c = conversion_create(ROWS, COLS, 0, 0, 0);
  if (!c) return EXIT_FAILURE;

  unsigned char chars[LEN * ROWS * COLS];
  signed char colors[LEN * ROWS * COLS];
  unsigned char curs[LEN * 2];
  int64_t timestamp[LEN];
  
  unsigned char inputs[LEN];


  if (conversion_load_ttyrec(c, fdopen(STDIN_FILENO, "r")) != 0)
    return EXIT_FAILURE;

  conversion_set_buffers(c, &chars[0], sizeof(chars), 
                         &colors[0], sizeof(colors),
                         &curs[0], LEN * 2,
                         &timestamp[0], LEN,
                         &inputs[0], LEN);

  for (int i = 0; conversion_convert_frames(c) == 1; ++i) {
    for (int j = 0; j < LEN; ++j) {
      printf("Conversion %d, iteration %d. Cursor at %hu,%hu\n", i, j,
             curs[2 * j + 0], curs[2 * j + 1]);
      printtime(timestamp[j]);
      for (size_t ro = 0; ro < c->rows; ++ro) {
        for (size_t co = 0; co < c->cols; ++co)
          putchar(chars[j * (c->rows * c->cols) + ro * c->cols + co]);
        putchar('\n');
      }
    }
    /* Reset buffers */
    conversion_set_buffers(c, &chars[0], sizeof(chars), 
                           &colors[0], sizeof(colors),
                           &curs[0], LEN * 2,
                           &timestamp[0], LEN,
                           &inputs[0], LEN);
  }
  conversion_close(c);
}
