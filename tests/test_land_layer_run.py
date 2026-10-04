"""A LAND quadrant with alpha layers and no base layer keeps its alpha layers.

See: docs/commentary/tes5_import_world.md#land-quadrants-without-a-base-layer
"""

import struct
from pathlib import Path

import pytest

from asset_convert.lod.terrain_lod_textures import decode_land_layers
from tes5_import.base.text_reader import (get_formid, get_int, get_str,
                                          parse_file_range)
from tes5_import.overrides.builder import apply_changes
from tes5_import.overrides.diff import diff_records
from tes5_import.record_types.world import build_land_layers, convert_LAND

REAL_LAND = (Path(__file__).resolve().parent.parent
             / 'export' / 'Oblivion.esm' / 'LAND.txt')
RANGE_BYTES = 16 * 1024 * 1024
RANGES_TRIED = 8

GRASS, DIRT, ROCK, MOSS = 0x0002506E, 0x0001C8BE, 0x0012FBE2, 0x0012FBE3


def _subs(blob):
    """[(tag, payload)] of a packed subrecord run."""
    out, off = [], 0
    while off + 6 <= len(blob):
        size = struct.unpack_from('<H', blob, off + 4)[0]
        out.append((blob[off:off + 4], blob[off + 6:off + 6 + size]))
        off += 6 + size
    return out


def _fid(tex):
    """A texture id as the importer writes it."""
    return get_formid({'Texture': '%08X' % tex}, 'Texture')


def _rec(layers):
    """An exported LAND from [(type, quadrant, texture, {pos: opacity})].

    Shaped like a real export block: ATXT.Layer counts from 0 per quadrant.
    """
    rec = {'Signature': 'LAND', 'FormID': '00008F26', 'RecordFlags': '0',
           'DATA.Flags': '31', 'LayerCount': str(len(layers))}
    seen = {}
    for i, (kind, quad, tex, points) in enumerate(layers):
        pfx = f'Layer[{i}]'
        rec[f'{pfx}.Type'] = kind
        key = 'BTXT' if kind == 'BASE' else 'ATXT'
        rec[f'{pfx}.{key}.Texture'] = '%08X' % tex
        rec[f'{pfx}.{key}.Quadrant'] = str(quad)
        if kind == 'ALPHA':
            rec[f'{pfx}.ATXT.Layer'] = str(seen.setdefault(quad, 0))
            seen[quad] += 1
            rec[f'{pfx}.VTXTCount'] = str(len(points))
            for j, (pos, opacity) in enumerate(sorted(points.items())):
                rec[f'{pfx}.VT[{j}].Pos'] = str(pos)
                rec[f'{pfx}.VT[{j}].Opacity'] = repr(opacity)
    return rec


def _packed(quads):
    """The expected run from [(quadrant, base or None, [(texture, {pos: opacity})])]."""
    out = b''
    for quad, base, alphas in quads:
        if base is not None:
            out += b'BTXT' + struct.pack('<HIBBxx', 8, _fid(base), quad, 0)
        for idx, (tex, points) in enumerate(alphas):
            out += b'ATXT' + struct.pack('<HIBBH', 8, _fid(tex), quad, 0, idx)
            out += b'VTXT' + struct.pack('<H', 8 * len(points)) + b''.join(
                struct.pack('<HHf', pos, 0, op)
                for pos, op in sorted(points.items()))
    return out


def _baseless(rec):
    """Quadrants of an exported LAND with a textured ALPHA layer and no BASE."""
    kinds = {}
    for i in range(get_int(rec, 'LayerCount')):
        pfx = f'Layer[{i}]'
        if get_str(rec, f'{pfx}.Type') == 'BASE':
            quad = get_int(rec, f'{pfx}.BTXT.Quadrant')
            kinds.setdefault(quad, set()).add('BASE')
        elif get_formid(rec, f'{pfx}.ATXT.Texture'):
            quad = get_int(rec, f'{pfx}.ATXT.Quadrant')
            kinds.setdefault(quad, set()).add('ALPHA')
    return {quad for quad, kind in kinds.items() if kind == {'ALPHA'}}


def _with_bases(rec, quads):
    """A copy of an exported LAND with a BASE layer appended for each of `quads`."""
    out = dict(rec)
    count = get_int(rec, 'LayerCount')
    for quad in sorted(quads):
        out[f'Layer[{count}].Type'] = 'BASE'
        out[f'Layer[{count}].BTXT.Texture'] = '%08X' % GRASS
        out[f'Layer[{count}].BTXT.Quadrant'] = str(quad)
        count += 1
    out['LayerCount'] = str(count)
    return out


def _alpha_as_under_a_base(rec, quads):
    """The run `rec` gets once `quads` have a BASE, with those BTXT taken out."""
    return [(tag, payload)
            for tag, payload in _subs(build_land_layers(_with_bases(rec, quads)))
            if not (tag == b'BTXT' and payload[4] in quads)]


def _real_baseless_lands():
    """[(record, baseless quadrants)] from the first export range holding any."""
    size = REAL_LAND.stat().st_size
    for start in range(0, min(size, RANGES_TRIED * RANGE_BYTES), RANGE_BYTES):
        records = parse_file_range((str(REAL_LAND), start, start + RANGE_BYTES))
        found = [(rec, _baseless(rec)) for rec in records]
        found = [(rec, quads) for rec, quads in found if quads]
        if found:
            return found
    return []


def test_baseless_quadrant_keeps_its_alpha_layers():
    """Quadrant 2 has two alpha layers and no BASE: both are written, no BTXT."""
    rec = _rec([('BASE', 0, GRASS, {}),
                ('ALPHA', 0, DIRT, {5: 0.5}),
                ('BASE', 1, GRASS, {}),
                ('ALPHA', 2, ROCK, {0: 1.0, 17: 0.75, 288: 0.25}),
                ('ALPHA', 2, DIRT, {3: 0.5}),
                ('BASE', 3, GRASS, {})])

    assert build_land_layers(rec) == _packed([
        (0, GRASS, [(DIRT, {5: 0.5})]),
        (1, GRASS, []),
        (2, None, [(ROCK, {0: 1.0, 17: 0.75, 288: 0.25}), (DIRT, {3: 0.5})]),
        (3, GRASS, [])])


def test_land_with_no_base_anywhere_still_writes_a_run():
    """No BASE in any quadrant: the LOD decoder reads the alpha layers back."""
    rec = _rec([('ALPHA', 0, ROCK, {0: 1.0, 288: 0.25}),
                ('ALPHA', 3, DIRT, {16: 0.5})])

    got = decode_land_layers(build_land_layers(rec))

    assert got['base'] == {} and sorted(got['alpha']) == [0, 3]
    [(tex, grid)] = got['alpha'][0]
    assert tex == _fid(ROCK)
    assert grid[0, 0] == 1.0 and grid[16, 16] == 0.25
    assert [tex for tex, _grid in got['alpha'][3]] == [_fid(DIRT)]


def test_quadrant_with_a_base_is_written_as_before():
    """Beside a baseless quadrant, a BASE quadrant keeps its merge, order and cap."""
    cover = [0.3, 0.9, 0.1, 0.8, 0.2, 0.7, 0.6]
    busy = [('ALPHA', 1, 0x2000 + i, {pos: c for pos in range(4)})
            for i, c in enumerate(cover)]
    rec = _rec([('BASE', 1, GRASS, {})] + busy + [
        ('ALPHA', 1, 0, {9: 1.0}),
        ('ALPHA', 1, 0x2000, {0: 0.1, 200: 0.4}),
        ('ALPHA', 2, MOSS, {7: 1.0})])

    merged = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 200: 0.4}
    kept = [(0x2001, 0.9), (0x2003, 0.8), (0x2005, 0.7), (0x2006, 0.6)]
    assert build_land_layers(rec) == _packed([
        (1, GRASS, [(tex, {pos: c for pos in range(4)}) for tex, c in kept]
         + [(0x2000, merged), (0x2004, {pos: 0.2 for pos in range(4)})]),
        (2, None, [(MOSS, {7: 1.0})])])


def test_baseless_quadrant_gets_the_layers_it_would_get_under_a_base():
    """Eight textures and a repeat, no BASE: same merge, order and cap as with one."""
    cover = [0.3, 0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.5]
    rec = _rec([('ALPHA', 2, 0x2000 + i, {pos: c for pos in range(4)})
                for i, c in enumerate(cover)]
               + [('ALPHA', 2, 0x2002, {100: 1.0})])

    got = _subs(build_land_layers(rec))

    assert [tag for tag, _payload in got].count(b'ATXT') > 1
    assert got == _alpha_as_under_a_base(rec, {2})


def test_override_carries_a_baseless_quadrant():
    """A plugin that paints a quadrant with alpha only gets it in its override."""
    master = _rec([('BASE', 0, GRASS, {})])
    plugin = _rec([('BASE', 0, GRASS, {}),
                   ('ALPHA', 1, ROCK, {40: 0.5})])

    out, _applied, unmapped = apply_changes(
        convert_LAND(master), diff_records(master, plugin), plugin, master)

    assert unmapped == set()
    assert _subs(out[24:])[1:] == _subs(_packed([
        (0, GRASS, []), (1, None, [(ROCK, {40: 0.5})])]))


@pytest.mark.skipif(not REAL_LAND.exists(),
                    reason='Oblivion.esm export not available')
def test_real_exported_land_keeps_its_baseless_quadrants():
    """Real LAND blocks, read as the importer reads them, keep such quadrants."""
    found = _real_baseless_lands()
    if not found:
        pytest.skip('no baseless quadrant in the first export ranges')
    for rec, quads in found:
        got = _subs(build_land_layers(rec))
        written = {payload[4] for tag, payload in got if tag == b'ATXT'}
        assert quads <= written, rec['FormID']
        assert got == _alpha_as_under_a_base(rec, quads), rec['FormID']
