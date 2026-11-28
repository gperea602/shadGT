"""FEAT-GT7-914. Convert a local two-stage Windows x64 GT7 capture to splash inputs.

Usage: python capture_seed.py CAPTURE_DIRECTORY NEW_ZIP
Supports the version-10 key ABI with no tessellation or generated helper code.
Refuses incomplete captures. The prepared store contains no host GPU binaries.
"""
import argparse
from hashlib import sha256
from pathlib import Path
import re
import struct
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from material_family import validate
from prepare_materials import PAIRS, RECORD, SERIAL, U32, hash_function, read_catalog


def combine(program, index):
    return (program ^ (index + 0x9e3779b9 + (program << 12) + (program >> 4))) & ((1 << 64) - 1)


def write_seed(root, output):
    if output.exists():
        raise FileExistsError(output)
    store = root / (SERIAL + '-direct-sets')
    manifest = root / (SERIAL + '.manifest')
    capture = root / (SERIAL + '.capture')
    digest = hash_function()
    version = (store / 'store-version.txt').read_bytes()
    _, close, switches = version.partition(b')')
    if not close or len(switches) > 4096:
        raise ValueError('Unrecognized store version')
    entries = {}
    for file in store.glob('*.spv'):
        match = re.fullmatch('0x([0-9a-f]{16})_([0-9]+)\\.spv', file.name)
        if not match:
            raise ValueError(f'Malformed shader name {file.name}')
        program, index = int(match[1], 16), int(match[2])
        perm = combine(program, index)
        if perm in entries:
            raise ValueError('Ambiguous permutation identity')
        entries[perm] = (program, index, file)
    recipes, files, checked = [], {}, {}

    def resolve(perm, stage):
        if (perm, stage) in checked:
            return checked[perm, stage]
        if perm not in entries:
            raise ValueError('Key references an unavailable shader')
        program, index, path = entries[perm]
        spv = path.read_bytes()
        validate(spv)
        meta = (store / f'0x{perm:016x}.meta').read_bytes()
        if len(meta) < 24 or struct.unpack_from('<QQ', meta, 8) != (perm, index):
            raise ValueError('Metadata names another permutation')
        variant = digest(spv)
        rec = manifest / f'{program:016x}-{variant:016x}.rec'
        data = rec.read_bytes()
        magic, ver, hw, sw, pgm, code_hash, words, spv_hash = RECORD.unpack_from(data)
        expected_hw = {0: 0, 3: 1}[stage]
        if (magic, ver, hw, sw, pgm, spv_hash) != (b'RYGTMAN1', 4, expected_hw, stage, program, variant):
            raise ValueError('Manifest identity or stage mismatch')
        code = capture / f'{program:016x}-{code_hash:016x}.code'
        raw = code.read_bytes()
        if not words or len(raw) != 4 * words or digest(raw) != code_hash:
            raise ValueError('Missing or mismatched guest program')
        files[rec.relative_to(root).as_posix()] = data
        files[code.relative_to(root).as_posix()] = raw
        checked[perm, stage] = (program, variant)
        return program, variant

    for key in sorted(store.glob('g_*.key')):
        data = key.read_bytes()
        # Confirm the bytewise hash used by std::hash<GraphicsPipelineKey>.
        # 376 bytes of key and the fixed vertex/multisampling support followed
        # by two empty size_t-prefixed helper vectors in this captured ABI.
        if (len(data) != 1624 or struct.unpack_from('<II',data) != (10,0)
                or data[-16:] != bytes(16) or key.stem != f'g_0x{digest(data[8:384]):016x}'):
            raise ValueError(f'Unsupported or incomplete graphics key {key.name}')
        stages = struct.unpack_from('<6Q',data,8)
        if not stages[0] or not stages[3] or any(stages[i] for i in (1,2,4,5)):
            raise ValueError('Expected only fragment and vertex stages')
        names = [resolve(perm, i) if perm else (0,0) for i,perm in enumerate(stages)]
        pairs = PAIRS.pack(*(v for name in names for v in name))
        recipes.append((pairs,data))
    if not recipes:
        raise ValueError('No graphic recipes')
    catalog = bytearray(b'RYGTRCP1' + U32.pack(len(switches)) + switches + U32.pack(len(recipes)))
    for pairs,data in recipes:
        catalog.extend(pairs + U32.pack(len(data)) + data)
    name = 'pipelines-materials-' + sha256(catalog).hexdigest()[:16] + '.rcp'
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        test = Path(work) / name
        test.write_bytes(catalog)
        if read_catalog(test) != (switches, recipes):
            raise ValueError('Recipe catalog round-trip failed')
        files[manifest.name + '/' + name] = bytes(catalog)
        pending = Path(work) / 'private.zip'
        with ZipFile(pending, 'w') as z:
            for path, data in sorted(files.items()):
                z.writestr(ZipInfo('shaders/' + path), data, compress_type=ZIP_DEFLATED)
        with ZipFile(pending) as z:
            if z.testzip() or set(z.namelist()) != {'shaders/' + path for path in files}:
                raise ValueError('Seed failed integrity check')
        # Open exclusively so a simultaneous run cannot replace a previous output.
        import os
        os.link(pending, output)
    return len(recipes), len(checked), name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(write_seed(args.capture, args.output))
