# SPDX-License-Identifier: GPL-2.0-or-later
"""FEAT-GT7-914. Select verified, complete graphics recipes for slow textured draws.

Usage: python prepare_materials.py LOG SHADERS CATALOG NEW_DIRECTORY
Output is a private startup input for a matching GT7 configuration. Keep the
local manifest and code capture alongside it when importing into an empty store.
"""
import argparse
from collections import Counter
import ctypes
import ctypes.util
import hashlib
import os
from pathlib import Path
import re
import struct
import tempfile

from probe import has_texture_sample


SERIAL = 'CUSA24769'
PAIRS = struct.Struct('<' + 'QQ' * 6)
RECORD = struct.Struct('<8sIIIQQIQ')
U32 = struct.Struct('<I')
EVENT = re.compile(r'Graphics pipeline ([^:]+): libraries ([\d.]+) ms '
                   r'\(pre-raster reused, fragment compiled\)')
STAGES = re.compile(r'(fs|vs)_0x([0-9a-fA-F]+)')


def slow_draws(log):
    pairs = Counter()
    with log.open(errors='replace') as lines:
        if not any('FEAT-SGT-903: startup queues drained' in line for line in lines):
            raise ValueError('No startup boundary')
        for line in lines:
            m = EVENT.search(line)
            if m and float(m[2]) >= 100:
                ids = {k: int(v, 16) for k, v in STAGES.findall(m[1])}
                if 'fs' in ids and 'vs' in ids:
                    pairs[ids['fs'], ids['vs']] += 1
    return pairs


def read_catalog(path):
    def take(stream, n):
        data = stream.read(n)
        if len(data) != n:
            raise ValueError('Truncated pipeline catalog')
        return data
    with path.open('rb') as fp:
        if take(fp, 8) != b'RYGTRCP1':
            raise ValueError('Unsupported recipe format')
        length, = U32.unpack(take(fp, 4))
        if length > 4096:
            raise ValueError('Invalid switch string')
        switches = take(fp, length)
        count, = U32.unpack(take(fp, 4))
        recipes = []
        for _ in range(count):
            names = take(fp, PAIRS.size)
            size, = U32.unpack(take(fp, 4))
            if size < 56 or size > 1 << 20:
                raise ValueError('Invalid pipeline size')
            recipes.append((names, take(fp, size)))
        if fp.read(1):
            raise ValueError('Unexpected trailing data')
    return switches, recipes


def hash_function():
    lib = ctypes.CDLL(ctypes.util.find_library('xxhash') or 'libxxhash.so.0')
    digest = lib.XXH3_64bits
    digest.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    digest.restype = ctypes.c_uint64
    return lambda data: digest(data, len(data))


def prepare(log, shaders, catalog, destination):
    if destination.exists():
        raise FileExistsError(destination)
    pairs = slow_draws(log)
    programs = {frag for frag, _ in pairs}
    switches, recipes = read_catalog(catalog)
    store = shaders / (SERIAL + '-direct-sets')
    version = (store / 'store-version.txt').read_bytes().partition(b')')
    if not version[1] or version[2] != switches:
        raise ValueError('Store settings differ from the recipe catalog')
    digest = hash_function()
    ids = set()
    for path in store.glob('*.spv'):
        m = re.fullmatch(r'0x([0-9a-fA-F]+)_\d+\.spv', path.name)
        if m and int(m[1], 16) in programs and has_texture_sample(path.read_bytes()):
            ids.add(int(m[1], 16))
    checked = {}
    def complete(pgm, variant, expected_stage):
        name = (pgm, variant, expected_stage)
        if name not in checked:
            rec = shaders / (SERIAL + '.manifest') / f'{pgm:016x}-{variant:016x}.rec'
            good = False
            if rec.is_file():
                with rec.open('rb') as stream:
                    header = stream.read(RECORD.size)
                if len(header) == RECORD.size:
                    magic, ver, hw, sw, prog, code_hash, size, spirv = RECORD.unpack(header)
                    if (magic == b'RYGTMAN1' and 1 <= ver <= 4 and sw == expected_stage
                            and prog == pgm and spirv == variant and size > 0):
                        code = shaders / (SERIAL + '.capture') / f'{pgm:016x}-{code_hash:016x}.code'
                        if code.is_file() and code.stat().st_size == 4 * size:
                            good = digest(code.read_bytes()) == code_hash
            checked[name] = good
        return checked[name]

    selected = []
    known_pairs = set()
    selected_ids = set()
    for names, blob in recipes:
        stages = tuple(struct.iter_unpack('<QQ', names))
        fs, vs = stages[0][0], stages[3][0]
        if fs not in ids or not vs or struct.unpack_from('<II', blob) != (10, 0):
            continue
        perm_hashes = struct.unpack_from('<6Q', blob, 8)
        if any((not perm and (p or v)) or (perm and not complete(p, v, i))
               for i, (perm, (p, v)) in enumerate(zip(perm_hashes, stages))):
            continue
        selected.append((names, blob))
        known_pairs.add((fs, vs))
        selected_ids.add(fs)
    if not selected:
        raise ValueError('No complete matching recipes')
    raw = bytearray(b'RYGTRCP1' + U32.pack(len(switches)) + switches + U32.pack(len(selected)))
    for names, blob in selected:
        raw.extend(names + U32.pack(len(blob)) + blob)
    name = 'pipelines-materials-' + hashlib.sha256(raw).hexdigest()[:16] + '.rcp'
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch) / 'stage'
        stage.mkdir()
        target = stage / name
        target.write_bytes(raw)
        if read_catalog(target) != (switches, selected):
            raise ValueError('Catalog round-trip failed')
        # Publish only a complete file in a new output directory.
        destination.mkdir()
        try:
            os.replace(target, destination / name)
        except OSError:
            destination.rmdir()
            raise
    return dict(log_events=sum(pairs.values()), captured_textured_events=sum(
        n for (fs, vs), n in pairs.items() if fs in ids), recipes=len(selected),
        unique_fragment_ids=len(selected_ids), events_with_same_fs_vs=sum(
            n for pair, n in pairs.items() if pair in known_pairs), catalog=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ('log', 'shaders', 'catalog', 'destination'):
        parser.add_argument(item, type=Path)
    print(prepare(**vars(parser.parse_args())))
