#version 450 core
// DIAG-GT7-905: offline experiment. The host admits only the exact sequence
// checked by classify() and supplies at most 16 little-endian words.
#ifdef VULKAN
layout(push_constant) uniform Program { uvec4 words[4]; };
#else
uniform uvec4 words[4];
#endif
layout(location = 0) out vec4 color;

uint readWord(uint index) {
    return words[index >> 2u][index & 3u];
}

void main() {
    uint regs[4];
    uint pc = 2u;
    while (pc < 15u) {
        uint op = readWord(pc++);
        if (op == 0xf8001c0fu) {
            uint sources = readWord(pc);
            color = vec4(unpackHalf2x16(regs[sources & 255u]),
                         unpackHalf2x16(regs[(sources >> 8u) & 255u]));
            return;
        }
        uint value = 0u;
        if ((op & 511u) == 255u) {
            value = readWord(pc++);
        }
        regs[(op >> 17u) & 3u] = value;
    }
    discard;
}
