/* Taken from stripgfx.c, dgamelaunch source */
/* Parts taken from drawing.c, nethack source */
/* Copyright (c) NetHack Development Team 1992.			  */
/* NetHack may be freely redistributed.  See license for details. */

#include <string.h>

#include "stripgfx.h"

#define MAXPCHARS 92

unsigned char gfx_dec_map[256];
unsigned char gfx_ibm_map[256];

/* clang-format off */
static unsigned char no_graphics[MAXPCHARS] = {
  ' ',                          /* 0 */
  /* stone */
  '|',                          /* vwall */
  '-',                          /* hwall */
  '-',                          /* tlcorn */
  '-',                          /* trcorn */
  '-',                          /* blcorn */
  '-',                          /* brcorn */
  '-',                          /* crwall */
  '-',                          /* tuwall */
  '-',                          /* tdwall */
  '|',                          /* 10 */
  /* tlwall */
  '|',                          /* trwall */
  '.',                          /* ndoor */
  '-',                          /* vodoor */
  '|',                          /* hodoor */
  '+',                          /* vcdoor */
  '+',                          /* hcdoor */
  '#',                          /* bars */
  '#',                          /* tree */
  '.',                          /* room */
  '#',                          /* 20 */
  /* dark corr */
  '#',                          /* lit corr */
  '<',                          /* upstair */
  '>',                          /* dnstair */
  '<',                          /* upladder */
  '>',                          /* dnladder */
  '_',                          /* altar */
  '|',                          /* grave */
  '\\',                         /* throne */
  '#',                          /* sink */
  '{',                          /* 30 */
  /* fountain */
  '}',                          /* pool */
  '.',                          /* ice */
  '}',                          /* lava */
  '.',                          /* vodbridge */
  '.',                          /* hodbridge */
  '#',                          /* vcdbridge */
  '#',                          /* hcdbridge */
  ' ',                          /* open air */
  '#',                          /* [part of] a cloud */
  '}',                          /* 40 */
  /* under water */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* 50 */
  /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '"',                          /* web */
  '^',                          /* trap */
  '^',                          /* 60 */
  /* trap */
  '^',                          /* trap */
  '^',                          /* trap */
  '|',                          /* vbeam */
  '-',                          /* hbeam */
  '\\',                         /* lslant */
  '/',                          /* rslant */
  '*',                          /* dig beam */
  '!',                          /* camera flash beam */
  ')',                          /* boomerang open left */
  '(',                          /* 70 */
  /* boomerang open right */
  '0',                          /* 4 magic shield symbols */
  '#',
  '@',
  '*',
  '/',                          /* swallow top left  */
  '-',                          /* swallow top center  */
  '\\',                         /* swallow top right  */
  '|',                          /* swallow middle left */
  '|',                          /* swallow middle right  */
  '\\',                         /* 80 */
  /* swallow bottom left  */
  '-',                          /* swallow bottom center */
  '/',                          /* swallow bottom right  */
  '/',                          /* explosion top left     */
  '-',                          /* explosion top center   */
  '\\',                         /* explosion top right    */
  '|',                          /* explosion middle left  */
  ' ',                          /* explosion middle center */
  '|',                          /* explosion middle right */
  '\\',                         /* explosion bottom left  */
  '-',                          /* 90 */
  /* explosion bottom center */
  '/'                           /* explosion bottom right */
};

static unsigned char ibm_graphics[MAXPCHARS] = {
/* 0*/ 0x00,
  0xb3,                         /* : meta-3, vertical rule */
  0xc4,                         /* : meta-D, horizontal rule */
  0xda,                         /* :  meta-Z, top left corner */
  0xbf,                         /* :  meta-?, top right corner */
  0xc0,                         /* :  meta-@, bottom left */
  0xd9,                         /* :  meta-Y, bottom right */
  0xc5,                         /* :  meta-E, cross */
  0xc1,                         /* :  meta-A, T up */
  0xc2,                         /* :  meta-B, T down */
  /*10 */ 0xb4,
  /* :  meta-4, T left */
  0xc3,                         /* :  meta-C, T right */
  0xfa,                         /* : meta-z, centered dot */
  0xfe,                         /* :  meta-~, small centered square */
  0xfe,                         /* :  meta-~, small centered square */
  0x00,
  0x00,
  240,                          /* :  equivalence symbol */
  241,                          /* :  plus or minus symbol */
  0xfa,                         /* :  meta-z, centered dot */
  /*20 */ 0xb0,
  /* :  meta-0, light shading */
  0xb1,                         /* : meta-1, medium shading */
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  /*30 */ 0xf4,
  /* :  meta-t, integral top half */
  0xf7,                         /* :  meta-w, approx. equals */
  0xfa,                         /* : meta-z, centered dot */
  0xf7,                         /* :  meta-w, approx. equals */
  0xfa,                         /* : meta-z, centered dot */
  0xfa,                         /* : meta-z, centered dot */
  0x00,
  0x00,
  0x00,
  0x00,
  /*40 */ 0xf7,
  /* : meta-w, approx. equals */
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
/*50*/ 0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
/*60*/ 0x00,
  0x00,
  0x00,
  0xb3,                         /* : meta-3, vertical rule */
  0xc4,                         /* : meta-D, horizontal rule */
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
/*70*/ 0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0xb3,                         /* : meta-3, vertical rule */
  0xb3,                         /* : meta-3, vertical rule */
/*80*/ 0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0xb3,                         /* :  meta-3, vertical rule */
  0x00,
  0xb3,                         /* :  meta-3, vertical rule */
  0x00,
/*90*/ 0x00,
  0x00
};

static unsigned char dec_graphics[MAXPCHARS] = {
/* 0*/ 0x00,
  0xf8,                         /* : meta-x, vertical rule */
  0xf1,                         /* : meta-q, horizontal rule */
  0xec,                         /* :  meta-l, top left corner */
  0xeb,                         /* :  meta-k, top right corner */
  0xed,                         /* :  meta-m, bottom left */
  0xea,                         /* :  meta-j, bottom right */
  0xee,                         /* :  meta-n, cross */
  0xf6,                         /* :  meta-v, T up */
  0xf7,                         /* :  meta-w, T down */
  /*10 */ 0xf5,
  /* :  meta-u, T left */
  0xf4,                         /* :  meta-t, T right */
  0xfe,                         /* : meta-~, centered dot */
  0xe1,                         /* :  meta-a, solid block */
  0xe1,                         /* :  meta-a, solid block */
  0x00,
  0x00,
  0xfb,                         /* :  meta-{, small pi */
  0xe7,                         /* :  meta-g, plus-or-minus */
  0xfe,                         /* :  meta-~, centered dot */
/*20*/ 0x00,
  0x00,
  0x00,
  0x00,
  0xf9,                         /* :  meta-y, greater-than-or-equals */
  0xfa,                         /* :  meta-z, less-than-or-equals */
  0x00,                         /* 0xc3, \E)3: meta-C, dagger */
  0x00,
  0x00,
  0x00,
  /*30 */ 0x00,
  /* 0xdb, \E)3: meta-[, integral top half */
  0xe0,                         /* :  meta-\, diamond */
  0xfe,                         /* : meta-~, centered dot */
  0xe0,                         /* :  meta-\, diamond */
  0xfe,                         /* : meta-~, centered dot */
  0xfe,                         /* : meta-~, centered dot */
  0x00,
  0x00,
  0x00,
  0x00,
  /*40 */ 0xe0,
  /* : meta-\, diamond */
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
/*50*/ 0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,                         /* 0xbd, \E)3: meta-=, int'l currency */
  0x00,
/*60*/ 0x00,
  0x00,
  0x00,
  0xf8,                         /* : meta-x, vertical rule */
  0xf1,                         /* : meta-q, horizontal rule */
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
/*70*/ 0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0x00,
  0xef,                         /* : meta-o, high horizontal line */
  0x00,
  0xf8,                         /* : meta-x, vertical rule */
  0xf8,                         /* : meta-x, vertical rule */
/*80*/ 0x00,
  0xf3,                         /* : meta-s, low horizontal line */
  0x00,
  0x00,
  0xef,                         /* :  meta-o, high horizontal line */
  0x00,
  0xf8,                         /* :  meta-x, vertical rule */
  0x00,
  0xf8,                         /* :  meta-x, vertical rule */
  0x00,
  /*90 */ 0xf3,
  /* :  meta-s, low horizontal line */
  0x00
};

// Unused Variable (--Werror) Kept for Info
//
// static unsigned char IBM_r_oc_syms[18] = {  /* a la EPYX Rogue */
// /* 0*/ '\0',
//   0x00,
//   0x18,                         /* weapon: up arrow */
//   /*  0x0a, */ 0x00,
//   /* armor:  Vert rect with o */
//   /*  0x09, */ 0x00,
//   /* ring:   circle with arrow */
//   /* 5 */ 0x0c,
//   /* amulet: "female" symbol */
//   0x00,
//   0x05,                         /* food:   club (as in cards) */
//   0xad,                         /* potion: upside down '!' */
//   0x0e,                         /* scroll: musical note */
// /*10*/ 0x00,
//   0xe7,                         /* wand:   greek tau */
//   0x0f,                         /* gold:   yes it's the same as gems */
//   0x0f,                         /* gems:   fancy '*' */
//   0x00,
// /*15*/ 0x00,
//   0x00,
//   0x00
// };

/* clang-format on */

void populate_gfx_arrays() {
  int i;


  memset(gfx_ibm_map, 0, 256);
  memset(gfx_dec_map, 0, 256);

  for (i = 0; i < MAXPCHARS; i++) {
    if (dec_graphics[i] && !(gfx_dec_map[dec_graphics[i]]))
      gfx_dec_map[dec_graphics[i] - 128] = no_graphics[i];
    if (ibm_graphics[i]) gfx_ibm_map[ibm_graphics[i]] = no_graphics[i];
  }

  /* Check. */
  /*
  for (i = 0; i < 255; i++)
  {
    if (gfx_dec_map[i] && gfx_ibm_map[i]) {
      fprintf(stderr,
              "Collision at %i: %c vs %c\n",
              i, gfx_dec_map[i], gfx_ibm_map[i]);
    }
  }
  */
}

unsigned char
strip_gfx(unsigned char inchar, int use_dec)
{
  if (use_dec && gfx_dec_map[inchar])
    return gfx_dec_map[inchar];
  if (gfx_ibm_map[inchar])
    return gfx_ibm_map[inchar];
  return inchar;
}
