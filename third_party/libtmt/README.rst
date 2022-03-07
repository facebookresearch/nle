
============================================
libtmt - a simple terminal emulation library
============================================

libtmt is the Tiny Mock Terminal Library.  It provides emulation of a classic
smart text terminal, by maintaining an in-memory screen image.  Sending text
and command sequences to libtmt causes it to update this in-memory image,
which can then be examined and rendered however the user sees fit.

The imagined primary goal for libtmt is to for terminal emulators and
multiplexers; it provides the terminal emulation layer for the `mtm`_
terminal multiplexer, for example. Other uses include screen-scraping and
automated test harnesses.

libtmt is similar in purpose to `libtsm`_, but considerably smaller (500
lines versus 6500 lines). libtmt is also, in this author's humble opinion,
considerably easier to use.

.. _`mtm`: https://github.com/deadpixi/mtm
.. _`libtsm`: https://www.freedesktop.org/wiki/Software/kmscon/libtsm/

Major Features and Advantages
=============================

Works Out-of-the-Box
    libtmt emulates a well-known terminal type (`ansi`), the definition of
    which has been in the terminfo database since at least 1995.  There's no
    need to install a custom terminfo entry.  There's no claiming to be an
    xterm but only emulating a small subset of its features. Any program
    using terminfo works automatically: this includes vim, emacs, mc,
    cmus, nano, nethack, ...

Portable
    Written in pure C99.
    Optionally, the POSIX-mandated `wcwidth` function can be used, which
    provides minimal support for combining characters.

Small
    Less than 500 lines of C, including comments and whitespace.

Free
    Released under a BSD-style license, free for commercial and
    non-commerical use, with no restrictions on source code release or
    redistribution.

Simple
    Only 8 functions to learn, and really you can get by with 6!

International
    libtmt internally uses wide characters exclusively, and uses your C
    library's multibyte encoding functions.
    This means that the library automatically supports any encoding that
    your operating system does.

How to Use libtmt
=================

libtmt is a single C file and a single header.  Just include these files
in your project and you should be good to go.

By default, libtmt uses only ISO standard C99 features,
but see `Compile-Time Options`_ below.

Example Code
------------

Below is a simple program fragment giving the flavor of libtmt.
Note that another good example is the `mtm`_ terminal multiplexer:

.. _`mtm`: https://github.com/deadpixi/mtm

.. code:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include "tmt.h"

    /* Forward declaration of a callback.
     * libtmt will call this function when the terminal's state changes.
     */
    void callback(tmt_msg_t m, TMT *vt, const void *a, void *p);

    int
    main(void)
    {
        /* Open a virtual terminal with 2 lines and 10 columns.
         * The first NULL is just a pointer that will be provided to the
         * callback; it can be anything. The second NULL specifies that
         * we want to use the default Alternate Character Set; this
         * could be a pointer to a wide string that has the desired
         * characters to be displayed when in ACS mode.
         */
        TMT *vt = tmt_open(2, 10, callback, NULL, NULL);
        if (!vt)
            return perror("could not allocate terminal"), EXIT_FAILURE;

        /* Write some text to the terminal, using escape sequences to
         * use a bold rendition.
         *
         * The final argument is the length of the input; 0 means that
         * libtmt will determine the length dynamically using strlen.
         */
        tmt_write(vt, "\033[1mhello, world (in bold!)\033[0m", 0);

        /* Writing input to the virtual terminal can (and in this case, did)
         * call the callback letting us know the screen was updated. See the
         * callback below to see how that works.
         */
        tmt_close(vt);
        return EXIT_SUCCESS;
    }

    void
    callback(tmt_msg_t m, TMT *vt, const void *a, void *p)
    {
        /* grab a pointer to the virtual screen */
        const TMTSCREEN *s = tmt_screen(vt);
        const TMTPOINT *c = tmt_cursor(vt);

        switch (m){
            case TMT_MSG_BELL:
                /* the terminal is requesting that we ring the bell/flash the
                 * screen/do whatever ^G is supposed to do; a is NULL
                 */
                printf("bing!\n");
                break;

            case TMT_MSG_UPDATE:
                /* the screen image changed; a is a pointer to the TMTSCREEN */
                for (size_t r = 0; r < s->nline; r++){
                    if (s->lines[r]->dirty){
                        for (size_t c = 0; c < s->ncol; c++){
                            printf("contents of %zd,%zd: %lc (%s bold)\n", r, c,
                                   s->lines[r]->chars[c].c,
                                   s->lines[r]->chars[c].a.bold? "is" : "is not");
                        }
                    }
                }

                /* let tmt know we've redrawn the screen */
                tmt_clean(vt);
                break;

            case TMT_MSG_ANSWER:
                /* the terminal has a response to give to the program; a is a
                 * pointer to a string */
                printf("terminal answered %s\n", (const char *)a);
                break;

            case TMT_MSG_MOVED:
                /* the cursor moved; a is a pointer to the cursor's TMTPOINT */
                printf("cursor is now at %zd,%zd\n", c->r, c->c);
                break;
        }
    }

Data Types and Enumerations
---------------------------

.. code:: c

    /* an opaque structure */
    typedef struct TMT TMT;

    /* possible messages sent to the callback */
    typedef enum{
        TMT_MSG_MOVED,  /* the cursor changed position       */
        TMT_MSG_UPDATE, /* the screen image changed          */
        TMT_MSG_ANSWER, /* the terminal responded to a query */
        TMT_MSG_BELL    /* the terminal bell was rung        */
    } tmt_msg_T;

    /* a callback for the library
     * m is one of the message constants above
     * vt is a pointer to the vt structure
     * r is NULL for TMT_MSG_BELL
     *   is a pointer to the cursor's TMTPOINT for TMT_MSG_MOVED
     *   is a pointer to the terminal's TMTSCREEN for TMT_MSG_UPDATE
     *   is a pointer to a string for TMT_MSG_ANSWER
     * p is whatever was passed to tmt_open (see below).
     */
    typedef void (*TMTCALLBACK)(tmt_msg_t m, struct TMT *vt,
                                const void *r, void *p);

    /* color definitions */
    typedef enum{
        TMT_COLOR_BLACK,
        TMT_COLOR_RED,
        TMT_COLOR_GREEN,
        TMT_COLOR_YELLOW,
        TMT_COLOR_BLUE,
        TMT_COLOR_MAGENTA,
        TMT_COLOR_CYAN,
        TMT_COLOR_WHITE,
        TMT_COLOR_DEFAULT /* whatever the host terminal wants it to mean */
    } tmt_color_t;

    /* graphical rendition */
    typedef struct TMTATTRS TMTATTRS;
    struct TMTATTRS{
        bool bold;      /* character is bold             */
        bool dim;       /* character is half-bright      */
        bool underline; /* character is underlined       */
        bool blink;     /* character is blinking         */
        bool reverse;   /* character is in reverse video */
        bool invisible; /* character is invisible        */
        tmt_color_t fg; /* character foreground color    */
        tmt_color_t bg; /* character background color    */
    };

    /* characters */
    typedef struct TMTCHAR TMTCHAR;
    struct TMTCHAR{
        wchar_t  c; /* the character */
        TMTATTRS a; /* its rendition */
    };

    /* a position on the screen; upper left corner is 0,0 */
    typedef struct TMTPOINT TMTPOINT;
    struct TMTPOINT{
        size_t r; /* row    */
        size_t c; /* column */
    };

    /* a line of characters on the screen;
     * every line is always as wide as the screen
     */
    typedef struct TMTLINE TMTLINE;
    struct TMTLINE{
        bool dirty;     /* line has changed since it was last drawn */
        TMTCHAR chars;  /* the contents of the line                 */
    };

    /* a virtual terminal screen image */
    typedef struct TMTSCREEN TMTSCREEN;
    struct TMTSCREEN{
        size_t nline;    /* number of rows          */
        size_t ncol;     /* number of columns       */
        TMTLINE **lines; /* the lines on the screen */
    };

Functions
---------

`TMT *tmt_open(size_t nrows, size_t ncols, TMTCALLBACK cb, VOID *p, const wchar *acs);`
    Creates a new virtual terminal, with `nrows` rows and `ncols` columns.
    The callback `cb` will be called on updates, and passed `p` as a final
    argument. See the definition of `tmt_msg_t` above for possible values
    of each argument to the callback.

    Terminals must have a size of at least two rows and two columns.

    `acs` specifies the characters to use when in Alternate Character Set
    (ACS) mode. The default string (used if `NULL` is specified) is::

         L"><^v#+:o##+++++~---_++++|<>*!fo"

    See `Alternate Character Set`_ for more information.

    Note that the callback must be ready to be called immediately, as
    it will be called after initialization of the terminal is done, but
    before the call to `tmt_open` returns.

`void tmt_close(TMT *vt)`
    Close and free all resources associated with `vt`.

`bool tmt_resize(TMT *vt, size_t nrows, size_t ncols)`
    Resize the virtual terminal to have `nrows` rows and `ncols` columns.
    The contents of the area in common between the two sizes will be preserved.

    Terminals must have a size of at least two rows and two columns.

    If this function returns false, the resize failed (only possible in
    out-of-memory conditions or invalid sizes). If this happens, the terminal
    is trashed and the only valid operation is the close the terminal.

`void tmt_write(TMT *vt, const char *s, size_t n);`
    Write the provided string to the terminal, interpreting any escape
    sequences contained threin, and update the screen image. The last
    argument is the length of the input. If set to 0, the length is
    determined using `strlen`.

    The terminal's callback function may be invoked one or more times before
    a call to this function returns.

    The string is converted internally to a wide-character string using the
    system's current multibyte encoding. Each terminal maintains a private
    multibyte decoding state, and correctly handles mulitbyte characters that
    span multiple calls to this function (that is, the final byte(s) of `s`
    may be a partial mulitbyte character to be completed on the next call).

`const TMTSCREEN *tmt_screen(const TMT *vt);`
    Returns a pointer to the terminal's screen image.

`const TMTPOINT *tmt_cursor(cosnt TMT *vt);`
    Returns a pointer to the terminal's cursor position.

`void tmt_clean(TMT *vt);`
    Call this after receiving a `TMT_MSG_UPDATE` or `TMT_MSG_MOVED` callback
    to let the library know that the program has handled all reported changes
    to the screen image.

`void tmt_reset(TMT *vt);`
    Resets the virtual terminal to its default state (colors, multibyte
    decoding state, rendition, etc).

Special Keys
------------

To send special keys to a program that is using libtmt for its display,
write one of the `TMT_KEY_*` strings to that program's standard input
(*not* to libtmt; it makes no sense to send any of these constants to
libtmt itself).

The following macros are defined, and are all constant strings:

- TMT_KEY_UP
- TMT_KEY_DOWN
- TMT_KEY_RIGHT
- TMT_KEY_LEFT
- TMT_KEY_HOME
- TMT_KEY_END
- TMT_KEY_INSERT
- TMT_KEY_BACKSPACE
- TMT_KEY_ESCAPE
- TMT_KEY_BACK_TAB
- TMT_KEY_PAGE_UP
- TMT_KEY_PAGE_DOWN
- TMT_KEY_F1 through TMT_KEY_F10

Note also that the classic PC console sent the enter key as
a carriage return, not a linefeed. Many programs don't care,
but some do.

Compile-Time Options
--------------------

There are two preprocessor macros that affect libtmt:

`TMT_INVALID_CHAR`
    Define this to a wide-character. This character will be added to
    the virtual display when an invalid multibyte character sequence
    is encountered.

    By default (if you don't define it as something else before compiling),
    this is `((wchar_t)0xfffd)`, which is the codepoint for the Unicode
    'REPLACEMENT CHARACTER'. Note that your system might not use Unicode,
    and its wide-character type might not be able to store a constant as
    large as `0xfffd`, in which case you'll want to use an alternative.

`TMT_HAS_WCWIDTH`
    By default, libtmt uses only standard C99 features.  If you define
    TMT_HAS_WCWIDTH before compiling, libtmt will use the POSIX `wcwidth`
    function to detect combining characters.

    Note that combining characters are still not handled particularly
    well, regardless of whether this was defined. Also note that what
    your C library's `wcwidth` considers a combining character and what
    the written language in question considers one could be different.

Alternate Character Set
-----------------------

The terminal can be switched to and from its "Alternate Character Set" (ACS)
using escape sequences. The ACS traditionally contained box-drawing and other
semigraphic characters.

The characters in the ACS are configurable at runtime, by passing a wide string
to `tmt_open`. The default if none is provided (i.e. the argument is `NULL`)
uses ASCII characters to approximate the traditional characters.

The string passed to `tmt_open` must be 31 characters long. The characters,
and their default ASCII-safe values, are in order:

- RIGHT ARROW ">"
- LEFT ARROW "<"
- UP ARROW "^"
- DOWN ARROW "v"
- BLOCK "#"
- DIAMOND "+"
- CHECKERBOARD "#"
- DEGREE "o"
- PLUS/MINUS "+"
- BOARD ":"
- LOWER RIGHT CORNER "+"
- UPPER RIGHT CORNER "+"
- UPPER LEFT CORNER "+"
- LOWER LEFT CORNER "+"
- CROSS "+"
- SCAN LINE 1 "~"
- SCAN LINE 3 "-"
- HORIZONTAL LINE "-"
- SCAN LINE 7 "-"
- SCAN LINE 9 "_"
- LEFT TEE "+"
- RIGHT TEE "+"
- BOTTOM TEE "+"
- TOP TEE "+"
- VERTICAL LINE "|"
- LESS THAN OR EQUAL "<"
- GREATER THAN OR EQUAL ">"
- PI "*"
- NOT EQUAL "!"
- POUND STERLING "f"
- BULLET "o"

If your system's wide character type's character set corresponds to the
Universal Character Set (UCS/Unicode), the following wide string is a
good option to use::

    L"→←↑↓■◆▒°±▒┘┐┌└┼⎺───⎽├┤┴┬│≤≥π≠£•"

**Note that multibyte decoding is disabled in ACS mode.** The traditional
implementations of the "ansi" terminal type (i.e. IBM PCs and compatibles)
had no concept of multibyte encodings and used the character codes
outside the ASCII range for various special semigraphic characters.
(Technically they had an entire alternate character set as well via the
code page mechanism, but that's beyond the scope of this explanation.)

The end result is that the terminfo definition of "ansi" sends characters
with the high bit set when in ACS mode. This breaks several multibyte
encoding schemes (including, most importantly, UTF-8).

As a result, libtmt does not attempt to decode multibyte characters in
ACS mode, since that would break the multibyte encoding, the semigraphic
characters, or both.

In general this isn't a problem, since programs explicitly switch to and
from ACS mode using escape sequences.

When in ACS mode, bytes that are not special members of the alternate
character set (that is, bytes not mapped to the string provided to
`tmt_open`) are passed unchanged to the terminal.

Supported Input and Escape Sequences
====================================

Internally libtmt uses your C library's/compiler's idea of a wide character
for all characters, so you should be able to use whatever characters you want
when writing to the virtual terminal (but see `Alternate Character Set`_).

The following escape sequences are recognized and will be processed
specially.

In the descriptions below, "ESC" means a literal escape character and "Ps"
means zero or more decimal numeric arguments separated by semicolons.
In descriptions "P1", "P2", etc, refer to the first parameter, second
parameter, and so on.  If a required parameter is omitted, it defaults
to the smallest meaningful value (zero if the command accepts zero as
an argument, one otherwise).  Any number of parameters may be passed,
but any after the first eight are ignored.

Unless explicitly stated below, cursor motions past the edges of the screen
are ignored and do not result in scrolling.  When characters are moved,
the spaces left behind are filled with blanks and any characters moved
off the edges of the screen are lost.

======================  ======================================================================
Sequence                Action
======================  ======================================================================
0x07 (Bell)             Callback with TMT_MSG_BELL
0x08 (Backspace)        Cursor left one cell
0x09 (Tab)              Cursor to next tab stop or end of line
0x0a (Carriage Return)  Cursor to first cell on this line
0x0d (Linefeed)         Cursor to same column one line down, scroll if needed
ESC H                   Set a tabstop in this column
ESC 7                   Save cursor position and current graphical state
ESC 8                   Restore saved cursor position and current graphical state
ESC c                   Reset terminal to default state
ESC [ Ps A              Cursor up P1 rows
ESC [ Ps B              Cursor down P1 rows
ESC [ Ps C              Cursor right P1 columns
ESC [ Ps D              Cursor left P1 columns
ESC [ Ps E              Cursor to first column of line P1 rows down from current
ESC [ Ps F              Cursor to first column of line P1 rows up from current
ESC [ Ps G              Cursor to column P1
ESC [ Ps d              Cursor to row P1
ESC [ Ps H              Cursor to row P1, column P2
ESC [ Ps f              Alias for ESC [ Ps H
ESC [ Ps I              Cursor to next tab stop
ESC [ Ps J              Clear screen
                        P1 == 0: from cursor to end of screen
                        P1 == 1: from beginning of screen to cursor
                        P1 == 2: entire screen
ESC [ Ps K              Clear line
                        P1 == 0: from cursor to end of line
                        P1 == 1: from beginning of line to cursor
                        P1 == 2: entire line
ESC [ Ps L              Insert P1 lines at cursor, scrolling lines below down
ESC [ Ps M              Delete P1 lines at cursor, scrolling lines below up
ESC [ Ps P              Delete P1 characters at cursor, moving characters to the right over
ESC [ Ps S              Scroll screen up P1 lines
ESC [ Ps T              Scroll screen down P1 lines
ESC [ Ps X              Erase P1 characters at cursor (overwrite with spaces)
ESC [ Ps Z              Go to previous tab stop
ESC [ Ps b              Repeat previous character P1 times
ESC [ Ps c              Callback with TMT_MSG_ANSWER "\033[?6c"
ESC [ Ps g              If P1 == 3, clear all tabstops
ESC [ Ps h              If P1 == 25, show the cursor (if it was hidden)
ESC [ Ps m              Change graphical rendition state; see below
ESC [ Ps l              If P1 == 25, hide the cursor
ESC [ Ps n              If P1 == 6, callback with TMT_MSG_ANSWER "\033[%d;%dR"
                        with cursor row, column
ESC [ Ps s              Alias for ESC 7
ESC [ Ps u              Alias for ESC 8
ESC [ Ps @              Insert P1 blank spaces at cursor, moving characters to the right over
======================  ======================================================================

For the `ESC [ Ps m` escape sequence above ("Set Graphic Rendition"),
up to eight parameters may be passed; the results are cumulative:

==============   =================================================
Rendition Code   Meaning
==============   =================================================
0                Reset all graphic rendition attributes to default
1                Bold
2                Dim (half bright)
4                Underline
5                Blink
7                Reverse video
8                Invisible
10               Leave ACS mode
11               Enter ACS mode
22               Bold off
23               Dim (half bright) off
24               Underline off
25               Blink off
27               Reverse video off
28               Invisible off
30               Foreground black
31               Foreground red
32               Foreground green
33               Foreground yellow
34               Foreground blue
35               Foreground magenta
36               Foreground cyan
37               Foreground white
39               Foreground default color
40               Background black
41               Background red
42               Background green
43               Background yellow
44               Background blue
45               Background magenta
46               Background cyan
47               Background white
49               Background default color
==============   =================================================

Other escape sequences are recognized but ignored.  This includes escape
sequences for switching out codesets (officially, all code sets are defined
as equivalent in libtmt), and the various "Media Copy" escape sequences
used to print output on paper (officially, there is no printer attached
to libtmt).

Additionally, "?" characters are stripped out of escape sequence parameter
lists for compatibility purposes.

Known Issues
============

- Combining characters are "handled" by ignoring them
  (when compiled with `TMT_HAS_WCWIDTH`) or by printing them separately.
- Double-width characters are rendered as single-width invalid
  characters.
- The documentation and error messages are available only in English.

Frequently Asked Questions
==========================

What programs work with libtmt?
-------------------------------

Pretty much all of them.  Any program that doesn't assume what terminal
it's running under should work without problem; this includes any program
that uses the terminfo, termcap, or (pd|n)?curses libraries.  Any program
that assumes it's running under some specific terminal might fail if its
assumption is wrong, and not just under libtmt.

I've tested quite a few applications in libtmt and they've worked flawlessly:
vim, GNU emacs, nano, cmus, mc (Midnight Commander), and others just work
with no changes.

What programs don't work with libtmt?
-------------------------------------

Breakage with libtmt is of two kinds: breakage due to assuming a terminal
type, and reduced functionality.

In all my testing, I only found one program that didn't work correctly by
default with libtmt: recent versions of Debian's `apt`_ assume a terminal
with definable scrolling regions to draw a fancy progress bar during
package installation.  Using apt in its default configuration in libtmt will
result in a corrupted display (that can be fixed by clearing the screen).

.. _`apt`: https://wiki.debian.org/Apt

In my honest opinion, this is a bug in apt: it shouldn't assume the type
of terminal it's running in.

The second kind of breakage is when not all of a program's features are
available.  The biggest missing feature here is mouse support: libtmt
doesn't, and probably never will, support mouse tracking.  I know of many
programs that *can* use mouse tracking in a terminal, but I don't know
of any that *require* it.  Most (if not all?) programs of this kind would
still be completely usable in libtmt.

License
-------

Copyright (c) 2017 Rob King
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
- Neither the name of the copyright holder nor the
  names of contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS,
COPYRIGHT HOLDERS, OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
