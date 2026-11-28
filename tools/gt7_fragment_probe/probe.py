"""DIAG-GT7-905. Compare a narrow fragment interpreter with local GT7 captures.

Run: python probe.py <shaders-directory> [--log stdout.log]
Reads captured material locally. It does not change the store or run the game.
"""
import argparse
import ctypes as ct
import ctypes.util
from collections import Counter, defaultdict
from pathlib import Path
import re
import struct


def classify(words):
    """Accept a constant-only GCN fragment ending in a full, packed MRT0 export."""
    if not 7 <= len(words) <= 16 or words[0] != 0xbeeb03ff:
        return False
    pc = 2
    defined = set()
    while pc < len(words) - 3:
        op = words[pc]
        pc += 1
        dest, source = (op >> 17) & 255, op & 511
        if op & 0xfe01fe00 != 0x7e000200 or dest >= 4 or source not in (128, 255):
            return False
        if source == 255:
            pc += 1
        defined.add(dest)
    if pc != len(words) - 3:
        return False
    export, sources, end = words[pc:]
    return (export == 0xf8001c0f and end == 0xbf810000 and sources >> 16 == 0
            and (sources & 255) in defined and (sources >> 8) in defined)


def unpack_words(blob):
    if len(blob) % 4:
        return ()
    return struct.unpack(f'<{len(blob) // 4}I', blob)


def reference_color(blob):
    """Evaluate only an exact straight-line subset of Giulio's translated SPIR-V.

    Reject any other operation in the entry point, including texture reads,
    extra outputs, swizzles that cannot be evaluated, and conditional control flow.
    Return the four independent floating-point output components for comparison.
    """
    words = unpack_words(blob)
    if len(words) < 5 or words[0] != 0x07230203:
        return None
    values, addresses, locations, outputs = {}, {}, {}, {}
    imports, entries, output_vars = set(), set(), set()
    active = ended = returned = False
    next_label = None
    pos = 5
    while pos < len(words):
        size, op = words[pos] >> 16, words[pos] & 65535
        if not size or pos + size > len(words):
            return None
        a = words[pos + 1:pos + size]
        pos += size
        if op == 11 and len(a) > 1 and struct.pack(f'<{len(a)-1}I', *a[1:]).rstrip(b'\0') == b'GLSL.std.450':
            imports.add(a[0])
        elif op == 15 and len(a) >= 3 and a[0] == 4:
            entries.add(a[1])
        elif op == 43 and len(a) == 3:
            values[a[1]] = a[2]
        elif op == 71 and len(a) == 3 and a[1] == 30:
            locations[a[0]] = a[2]
        elif op == 59 and len(a) >= 3 and a[2] == 3:
            output_vars.add(a[1])
        elif op == 54:
            if active or ended or len(a) != 4 or entries != {a[1]}:
                return None
            active = True
        elif active:
            if (returned and op != 56) or (next_label is not None and op != 248):
                return None
            if op == 248 and len(a) == 1 and (next_label is None or a[0] == next_label):
                next_label = None
            elif op == 12 and len(a) == 5 and a[2] in imports and a[3] == 62 and type(values.get(a[4])) is int:
                values[a[1]] = struct.unpack('<ee', struct.pack('<I', values[a[4]]))
            elif op == 81 and len(a) == 4 and isinstance(values.get(a[2]), tuple) and a[3] < 2:
                values[a[1]] = values[a[2]][a[3]]
            elif op == 65 and len(a) == 4 and a[2] in output_vars and locations.get(a[2]) == 0:
                component = values.get(a[3])
                if component not in (0, 1, 2, 3):
                    return None
                addresses[a[1]] = component
            elif op == 62 and len(a) == 2 and a[0] in addresses and isinstance(values.get(a[1]), float):
                outputs[addresses[a[0]]] = values[a[1]]
            elif op == 249 and len(a) == 1:
                next_label = a[0]
            elif op == 253 and not a:
                returned = True
            elif op == 56 and not a and returned:
                active, ended = False, True
            else:
                return None
    if ended and len(output_vars) == 1 and len(outputs) == 4:
        return tuple(outputs[i] for i in range(4))
    return None


def cases(root):
    base = 'CUSA24769'
    capture = root / (base + '.capture')
    store = root / (base + '-direct-sets')
    manifest = root / (base + '.manifest')
    if not all(d.is_dir() for d in (capture, store, manifest)):
        raise ValueError('Expected a GT7 capture, direct-sets store and manifest')
    records = {}
    for record in manifest.glob('*.rec'):
        data = record.read_bytes()
        if len(data) < 48 or data[:8] != b'RYGTMAN1':
            continue
        ver, hw, sw, pgm, code_hash, size, spv_hash = struct.unpack_from('<IIIQQIQ', data, 8)
        if ver in (1, 2, 3, 4) and hw == sw == 0:
            records.setdefault((pgm, code_hash, size), set()).add(spv_hash)

    xxh = ct.CDLL(ctypes.util.find_library('xxhash') or 'libxxhash.so.0').XXH3_64bits
    xxh.argtypes = [ct.c_void_p, ct.c_size_t]
    xxh.restype = ct.c_uint64
    stats = Counter()
    stage_ids, accepted, verified = set(), set(), []
    for (pgm, code_hash, size), translations in sorted(records.items()):
        file = capture / f'{pgm:016x}-{code_hash:016x}.code'
        if not file.is_file():
            continue
        raw = file.read_bytes()
        if len(raw) != size * 4 or xxh(raw, len(raw)) != code_hash:
            continue
        stage_ids.add(pgm)
        if size > 16:
            continue
        words = unpack_words(raw)
        if not classify(words):
            continue
        accepted.add(pgm)
        for translated in sorted(store.glob(f'0x{pgm:016x}_*.spv')):
            blob = translated.read_bytes()
            if xxh(blob, len(blob)) not in translations:
                continue
            stats['matched translations'] += 1
            color = reference_color(blob)
            if color is not None:
                verified.append((pgm, words, color))
    stats['captured fragment program IDs'] = len(stage_ids)
    stats['program IDs with admitted GCN'] = len(accepted)
    stats['comparable translations'] = len(verified)
    return stats, verified


def has_texture_sample(blob):
    """Check the fragment module for an OpImageSample* instruction (SPIR-V 87..94)."""
    words = unpack_words(blob)
    if len(words) < 5 or words[0] != 0x07230203:
        return None
    pos = 5
    fragment = sampled = False
    while pos < len(words):
        size, op = words[pos] >> 16, words[pos] & 65535
        if not size or pos + size > len(words):
            return None
        if op == 15 and size >= 4 and words[pos + 1] == 4:
            fragment = True
        if 87 <= op <= 94:
            sampled = True
        pos += size
    return sampled if fragment else None


def profile_session(path, root, verified):
    hits = {pgm for pgm, _, _ in verified}
    result = Counter({'post-start logged library builds': 0, 'logged library builds >=100ms': 0,
                      'logged builds with comparable fragment ID': 0,
                      '>=100ms with comparable fragment ID': 0})
    slow_fragments = Counter()
    pattern = re.compile(r'Graphics pipeline fs_0x([0-9a-fA-F]+)[^:]*: libraries ([0-9.]+) ms '
                         r'\(pre-raster (compiled|reused), fragment (compiled|reused)\)')
    with path.open(errors='replace') as lines:
        if not any('FEAT-SGT-903: startup queues drained' in line for line in lines):
            raise ValueError('Missing startup boundary in supplied log')
        for line in lines:
            m = pattern.search(line)
            if not m:
                continue
            pgm, ms = int(m[1], 16), float(m[2])
            result['post-start logged library builds'] += 1
            if pgm in hits:
                result['logged builds with comparable fragment ID'] += 1
            if ms < 100:
                continue
            result['logged library builds >=100ms'] += 1
            if pgm in hits:
                result['>=100ms with comparable fragment ID'] += 1
            if m[3] == 'reused' and m[4] == 'compiled':
                slow_fragments[pgm] += 1

    variants = defaultdict(list)
    for file in (root / 'CUSA24769-direct-sets').glob('*.spv'):
        match = re.fullmatch(r'0x([0-9a-fA-F]+)_\d+\.spv', file.name)
        if match and (pgm := int(match[1], 16)) in slow_fragments:
            variants[pgm].append(has_texture_sample(file.read_bytes()))
    result['>=100ms with pre-raster reused and fragment compiled'] = sum(slow_fragments.values())
    result['of those, events with a fragment ID in captured SPIR-V'] = sum(
        slow_fragments[p] for p in variants)
    result['of those, events with sampling in every captured variant'] = sum(
        slow_fragments[p] for p, samples in variants.items() if all(v is True for v in samples))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    stats, verified = cases(args.root)
    if args.log:
        stats.update(profile_session(args.log, args.root, verified))
    for label, count in stats.items():
        print(f'{label}: {count}')
