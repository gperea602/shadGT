"""FEAT-GT7-914: build one offline fragment module from two complete captured variants.

The experimental layout requires a fragment push range through byte 123. Outputs
are private lab inputs. Production pipeline layouts use the original 120 bytes.
"""
import argparse
import ctypes as ct
import ctypes.util
from pathlib import Path
import struct


def pack(words):
    return struct.pack(f'<{len(words)}I', *words)


def decode(data):
    if len(data) < 20 or len(data) % 4:
        raise ValueError('Invalid module size')
    words = struct.unpack(f'<{len(data)//4}I', data)
    if words[0] != 0x07230203:
        raise ValueError('Expected SPIR-V')
    pos, inst = 5, []
    while pos < len(words):
        count, op = divmod(words[pos], 65536)
        if not count or pos + count > len(words):
            raise ValueError('Malformed instruction')
        # Ignore debug names and strings. Validate the stripped baseline too,
        # rejecting modules whose instructions reference an omitted string.
        if op not in (5, 6, 7):
            inst.append((op, words[pos+1:pos+count]))
        pos += count
    return words[:5], inst


def encode(header, inst):
    return pack(header) + b''.join(pack(((len(args)+1) << 16 | op, *args)) for op,args in inst)


def only(values, label):
    values = list(values)
    if len(values) != 1:
        raise ValueError(f'Expected one {label}, found {len(values)}')
    return values[0]


def validate(blob):
    lib = ct.CDLL(ctypes.util.find_library('SPIRV-Tools-shared')
                  or ctypes.util.find_library('shaderc') or 'libshaderc.so.1')
    def fn(name, result, *args):
        f = getattr(lib, name)
        f.restype, f.argtypes = result, args
        return f
    ptr, u32 = ct.c_void_p, ct.c_uint32
    env = ct.c_int()
    if not fn('spvParseTargetEnv', ct.c_bool, ct.c_char_p, ct.POINTER(ct.c_int))(b'vulkan1.3', ct.byref(env)):
        raise RuntimeError('Validator lacks Vulkan 1.3')
    context = fn('spvContextCreate', ptr, ct.c_int)(env)
    if not context:
        raise RuntimeError('SPIR-V Tools context unavailable')
    diag = ptr()
    try:
        words = (u32 * (len(blob)//4)).from_buffer_copy(blob)
        status = fn('spvValidateBinary', ct.c_int, ptr, ct.POINTER(u32), ct.c_size_t, ct.POINTER(ptr))(
            context, words, len(words), ct.byref(diag))
        if status:
            if diag:
                fn('spvDiagnosticPrint', ct.c_int, ptr)(diag)
            raise ValueError(f'Vulkan 1.3 validation failed ({status})')
    finally:
        if diag:
            fn('spvDiagnosticDestroy', None, ptr)(diag)
        fn('spvContextDestroy', None, ptr)(context)


def family(left, right):
    for module in (left, right):
        validate(module)
    header, a = decode(left)
    other_header, b = decode(right)
    if header != other_header or len(a) != len(b):
        raise ValueError('Modules have different structure')
    validate(encode(header, a))
    validate(encode(header, b))
    idx = only((i for i,x in enumerate(a) if x != b[i]), 'instruction difference')
    op, literal = a[idx]
    if op != 43 or b[idx][0] != op or len(literal) != 3 or literal[:2] != b[idx][1][:2]:
        raise ValueError('Difference is not a scalar constant')
    f32, ident, first_value = literal
    if (22, (f32, 32)) not in a or not any(op == 15 and ar[0] == 4 for op,ar in a):
        raise ValueError('Expected a fragment shader and a 32-bit float')
    # No use in decorations, constant expressions or other instructions. Treat
    # ambiguous literal/ID collisions conservatively and reject those modules.
    use = only((i for i,(op,ar) in enumerate(a) if i != idx and ident in ar), 'use of the constant')
    if a[use][0] != 133 or len(a[use][1]) != 4 or ident not in a[use][1][2:]:
        raise ValueError('Expected one scalar FMul operand')
    if a[use][1][0] != f32 or a[use][1].count(ident) != 1:
        raise ValueError('Unsupported multiply shape')
    u32 = only((ar[0] for op,ar in a if op == 21 and ar[1:] == (32, 0)), 'uint32 type')
    push_ptr, push_var, _ = only((ar for op,ar in a if op == 59 and ar[2:] == (9,)), 'push block')
    block = only((ar[2] for op,ar in a if op == 32 and ar[:2] == (push_ptr, 9)), 'block type')
    si = only((i for i,(op,ar) in enumerate(a) if op == 30 and ar[0] == block), 'block declaration')
    members = a[si][1][1:]
    offsets = dict((ar[1], ar[3]) for op,ar in a if op == 72 and ar[0] == block and ar[2] == 35)
    types = {ar[0]: ar[1:] for op,ar in a if op == 23}
    if ([offsets.get(i) for i in range(len(members))] != [0,4,8,12,16,32,48,64,80,96,112]
            or members[:4] != (f32,)*4 or any(types.get(t) != (u32,4) for t in members[4:10])
            or types.get(members[-1]) != (u32,2)):
        raise ValueError('Unknown push block ABI')
    old_bound = header[3]
    float_ptr, member_idx, address, value = range(old_bound, old_bound+4)
    marker = next(i for i,(op,_) in enumerate(a) if op == 54)
    annotation = next(i for i,(op,ar) in enumerate(a) if op == 72 and ar[0] == block)
    out = []
    for i,(op,ar) in enumerate(a):
        if i == annotation:
            out.append((72,(block,len(members),35,120)))
        if i == si:
            ar += (f32,)
        if i == marker:
            out.extend([(32,(float_ptr,9,f32)), (43,(u32,member_idx,len(members)))])
        if i == idx:
            # Keep the original ID in the definition section for the round-trip
            # check. No executable instruction reads it in the shared module.
            ar = (f32,ident,0)
        if i == use:
            out.extend([(65,(float_ptr,address,push_var,member_idx)), (61,(f32,value,address))])
            ar = tuple(value if v == ident else v for v in ar)
        out.append((op,ar))
    h = (*header[:3],old_bound+4,header[4])
    shared = encode(h,out)
    validate(shared)

    # Independently fold the exact load from byte 120 and remove just the added ABI.
    # Every executable instruction of each captured shader must be recovered.
    words = (first_value, b[idx][1][2])
    for baseline, bits in zip((left,right), words):
        _, emitted = decode(shared)
        restored = []
        for op,ar in emitted:
            if (op,ar) in {(72,(block,len(members),35,120)),(32,(float_ptr,9,f32)),
                           (43,(u32,member_idx,len(members))),
                           (65,(float_ptr,address,push_var,member_idx)),(61,(f32,value,address))}:
                continue
            if op == 30 and ar[0] == block:
                ar = ar[:-1]
            if op == 43 and ar[1] == ident:
                ar = (f32,ident,bits)
            if op == 133:
                ar = tuple(ident if v == value else v for v in ar)
            restored.append((op,ar))
        if encode(header,restored) != encode(*decode(baseline)):
            raise ValueError('Specialized module differs from captured translation')
    return shared, words, sum(87 <= op <= 94 for op,ar in out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('left','right','output'):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    shared, values, samples = family(args.left.read_bytes(), args.right.read_bytes())
    with args.output.open('xb') as f:
        f.write(shared)
    print(f'Validated shared module, {samples} texture sampling instructions, '
          f'pixel-stage push values {[hex(v) for v in values]} at offset 120')
