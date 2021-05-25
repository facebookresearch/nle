# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum

import numpy as np

from nle.nethack import *  # noqa: F403

# flake8: noqa: F405

# TODO: import this from NLE again
NUM_OBJECTS = 453
MAXEXPCHARS = 9


class GlyphGroup(enum.IntEnum):
    # See display.h in NetHack.
    MON = 0
    PET = 1
    INVIS = 2
    DETECT = 3
    BODY = 4
    RIDDEN = 5
    OBJ = 6
    CMAP = 7
    EXPLODE = 8
    ZAP = 9
    SWALLOW = 10
    WARNING = 11
    STATUE = 12


def id_pairs_table():
    """Returns a lookup table for glyph -> NLE id pairs."""
    table = np.zeros([MAX_GLYPH, 2], dtype=np.int16)

    num_nle_ids = 0

    for glyph in range(GLYPH_MON_OFF, GLYPH_PET_OFF):
        table[glyph] = (glyph, GlyphGroup.MON)
        num_nle_ids += 1

    for glyph in range(GLYPH_PET_OFF, GLYPH_INVIS_OFF):
        table[glyph] = (glyph - GLYPH_PET_OFF, GlyphGroup.PET)

    for glyph in range(GLYPH_INVIS_OFF, GLYPH_DETECT_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.INVIS)
        num_nle_ids += 1

    for glyph in range(GLYPH_DETECT_OFF, GLYPH_BODY_OFF):
        table[glyph] = (glyph - GLYPH_DETECT_OFF, GlyphGroup.DETECT)

    for glyph in range(GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF):
        table[glyph] = (glyph - GLYPH_BODY_OFF, GlyphGroup.BODY)

    for glyph in range(GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF):
        table[glyph] = (glyph - GLYPH_RIDDEN_OFF, GlyphGroup.RIDDEN)

    for glyph in range(GLYPH_OBJ_OFF, GLYPH_CMAP_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.OBJ)
        num_nle_ids += 1

    for glyph in range(GLYPH_CMAP_OFF, GLYPH_EXPLODE_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.CMAP)
        num_nle_ids += 1

    for glyph in range(GLYPH_EXPLODE_OFF, GLYPH_ZAP_OFF):
        id_ = num_nle_ids + (glyph - GLYPH_EXPLODE_OFF) // MAXEXPCHARS
        table[glyph] = (id_, GlyphGroup.EXPLODE)

    num_nle_ids += EXPL_MAX

    for glyph in range(GLYPH_ZAP_OFF, GLYPH_SWALLOW_OFF):
        id_ = num_nle_ids + (glyph - GLYPH_ZAP_OFF) // 4
        table[glyph] = (id_, GlyphGroup.ZAP)

    num_nle_ids += NUM_ZAP

    for glyph in range(GLYPH_SWALLOW_OFF, GLYPH_WARNING_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.SWALLOW)
    num_nle_ids += 1

    for glyph in range(GLYPH_WARNING_OFF, GLYPH_STATUE_OFF):
        table[glyph] = (num_nle_ids, GlyphGroup.WARNING)
        num_nle_ids += 1

    for glyph in range(GLYPH_STATUE_OFF, MAX_GLYPH):
        table[glyph] = (glyph - GLYPH_STATUE_OFF, GlyphGroup.STATUE)

    return table


def id_pairs_func(glyph):
    result = glyph_to_mon(glyph)
    if result != NO_GLYPH:
        return result
    if glyph_is_invisible(glyph):
        return NUMMONS
    if glyph_is_body(glyph):
        return glyph - GLYPH_BODY_OFF

    offset = NUMMONS + 1

    # CORPSE handled by glyph_is_body; STATUE handled by glyph_to_mon.
    result = glyph_to_obj(glyph)
    if result != NO_GLYPH:
        return result + offset
    offset += NUM_OBJECTS

    # I don't understand glyph_to_cmap and/or the GLYPH_EXPLODE_OFF definition
    # with MAXPCHARS - MAXEXPCHARS.
    if GLYPH_CMAP_OFF <= glyph < GLYPH_EXPLODE_OFF:
        return glyph - GLYPH_CMAP_OFF + offset
    offset += MAXPCHARS - MAXEXPCHARS

    if GLYPH_EXPLODE_OFF <= glyph < GLYPH_ZAP_OFF:
        return (glyph - GLYPH_EXPLODE_OFF) // MAXEXPCHARS + offset
    offset += EXPL_MAX

    if GLYPH_ZAP_OFF <= glyph < GLYPH_SWALLOW_OFF:
        return ((glyph - GLYPH_ZAP_OFF) >> 2) + offset
    offset += NUM_ZAP

    if GLYPH_SWALLOW_OFF <= glyph < GLYPH_WARNING_OFF:
        return offset
    offset += 1

    result = glyph_to_warning(glyph)
    if result != NO_GLYPH:
        return result + offset
