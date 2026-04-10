
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1
#if defined(DATA_A_F32)
layout (binding = 1) readonly buffer K_PACKED {vec4 k_data_packed[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED {vec4 v_data_packed[];} v_packed;
#elif defined(A_TYPE_PACKED16)
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#if defined(DATA_A_F32)
#undef BLOCK_SIZE
#define BLOCK_SIZE 4
#define BLOCK_BYTE_SIZE 16

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // iqs is currently always zero in the flash attention shaders
    if (binding_idx == BINDING_IDX_K) {
        return FLOAT_TYPEV4(k_packed.k_data_packed[a_offset + ib]);
    } else {
        return FLOAT_TYPEV4(v_packed.v_data_packed[a_offset + ib]);
    }
}
#endif

#if defined(DATA_A_Q4_0)
#define BLOCK_BYTE_SIZE 18
#elif defined(DATA_A_Q4_1)
#define BLOCK_BYTE_SIZE 20
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q4_1
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * nibbles + FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));
#endif
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q4_1
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * nibbles + FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));
#endif
    }
}
#endif

#if defined(DATA_A_Q5_0)
#define BLOCK_BYTE_SIZE 22
#elif defined(DATA_A_Q5_1)
#define BLOCK_BYTE_SIZE 24
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

#ifdef DATA_A_Q5_1
        uint qh = k_packed.k_data_packed16[a_offset + ib].qh;
#else
        uint qh = uint(k_packed.k_data_packed16[a_offset + ib].qh[0]) | (uint(k_packed.k_data_packed16[a_offset + ib].qh[1]) << 16);
#endif
        FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs) & 1, (qh >> (iqs + 1)) & 1, (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1) * FLOAT_TYPE(16.0f);

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q5_1
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles + hb) + FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));
#endif
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

#ifdef DATA_A_Q5_1
        uint qh = v_packed.v_data_packed16[a_offset + ib].qh;
#else
        uint qh = uint(v_packed.v_data_packed16[a_offset + ib].qh[0]) | (uint(v_packed.v_data_packed16[a_offset + ib].qh[1]) << 16);
#endif
        FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs) & 1, (qh >> (iqs + 1)) & 1, (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1) * FLOAT_TYPE(16.0f);

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q5_1
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles + hb) + FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));
#endif
    }
}
#endif


#if defined(DATA_A_IQ4_NL)
#define BLOCK_BYTE_SIZE 18

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(
            kvalues_iq4nl[vui_lo & 0xF],
            kvalues_iq4nl[(vui_lo >> 8) & 0xF],
            kvalues_iq4nl[vui_hi & 0xF],
            kvalues_iq4nl[(vui_hi >> 8) & 0xF]);
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(
            kvalues_iq4nl[vui_lo & 0xF],
            kvalues_iq4nl[(vui_lo >> 8) & 0xF],
            kvalues_iq4nl[vui_hi & 0xF],
            kvalues_iq4nl[(vui_hi >> 8) & 0xF]);
    }
}
#endif
#if defined(DATA_A_Q8_0)
#define BLOCK_BYTE_SIZE 34
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    } else {
        const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    }
}
#endif

// ============================================================
// TurboQuant dequantize4 implementations
// Byte access via raw uint[] buffers declared in types.glsl.
// ============================================================

#if defined(DATA_A_TURBO3_0)
// block_turbo3_0: 32 values, 14 bytes
//   bytes 0-1: float16 norm
//   bytes 2-9: qs[8]    lower 2 bits of 3-bit index, 4 per byte
//   bytes 10-13: signs[4] upper 1 bit of index, 8 per byte
#define BLOCK_BYTE_SIZE 14

const float CENTROIDS_3BIT[8] = float[](
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685);

uint turbo3_byte_k(uint byte_abs) {
    return (k_turbo3_raw[byte_abs >> 2u] >> ((byte_abs & 3u) << 3u)) & 0xFFu;
}
uint turbo3_byte_v(uint byte_abs) {
    return (v_turbo3_raw[byte_abs >> 2u] >> ((byte_abs & 3u) << 3u)) & 0xFFu;
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // bi = absolute block index; block starts at byte bi*14
    uint bi = a_offset + ib;
    uint base = bi * 14u;

    // Extract fp16 norm from bytes 0-1 of the block
    uint nb0, nb1;
    if (binding_idx == BINDING_IDX_K) {
        nb0 = turbo3_byte_k(base);
        nb1 = turbo3_byte_k(base + 1u);
    } else {
        nb0 = turbo3_byte_v(base);
        nb1 = turbo3_byte_v(base + 1u);
    }
    float norm = unpackHalf2x16(nb0 | (nb1 << 8u)).x;

    // iqs is always a multiple of 4; all 4 elements share one qs byte and one signs byte
    // qs byte is at block offset 2 + iqs/4; signs byte at block offset 10 + iqs/8
    uint qs_off    = base + 2u + iqs / 4u;
    uint signs_off = base + 10u + iqs / 8u;
    uint qs_byte, signs_byte;
    if (binding_idx == BINDING_IDX_K) {
        qs_byte    = turbo3_byte_k(qs_off);
        signs_byte = turbo3_byte_k(signs_off);
    } else {
        qs_byte    = turbo3_byte_v(qs_off);
        signs_byte = turbo3_byte_v(signs_off);
    }

    // iqs%8 is the bit offset in the signs byte for element iqs
    uint sign_shift = iqs & 7u;

    float r0 = norm * CENTROIDS_3BIT[( qs_byte        & 0x3u) | (((signs_byte >> (sign_shift + 0u)) & 1u) << 2u)];
    float r1 = norm * CENTROIDS_3BIT[((qs_byte >> 2u) & 0x3u) | (((signs_byte >> (sign_shift + 1u)) & 1u) << 2u)];
    float r2 = norm * CENTROIDS_3BIT[((qs_byte >> 4u) & 0x3u) | (((signs_byte >> (sign_shift + 2u)) & 1u) << 2u)];
    float r3 = norm * CENTROIDS_3BIT[((qs_byte >> 6u) & 0x3u) | (((signs_byte >> (sign_shift + 3u)) & 1u) << 2u)];

    return FLOAT_TYPEV4(r0, r1, r2, r3);
}
#endif // DATA_A_TURBO3_0

#if defined(DATA_A_TURBO2_0)
// block_turbo2_0: 32 values, 10 bytes
//   bytes 0-1: float16 norm
//   bytes 2-9: qs[8]  2-bit indices, 4 per byte
#define BLOCK_BYTE_SIZE 10

const float CENTROIDS_2BIT[4] = float[](-0.133462, -0.039994, 0.039994, 0.133462);

uint turbo2_byte_k(uint byte_abs) {
    return (k_turbo2_raw[byte_abs >> 2u] >> ((byte_abs & 3u) << 3u)) & 0xFFu;
}
uint turbo2_byte_v(uint byte_abs) {
    return (v_turbo2_raw[byte_abs >> 2u] >> ((byte_abs & 3u) << 3u)) & 0xFFu;
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    uint bi = a_offset + ib;
    uint base = bi * 10u;

    uint nb0, nb1;
    if (binding_idx == BINDING_IDX_K) {
        nb0 = turbo2_byte_k(base);
        nb1 = turbo2_byte_k(base + 1u);
    } else {
        nb0 = turbo2_byte_v(base);
        nb1 = turbo2_byte_v(base + 1u);
    }
    float norm = unpackHalf2x16(nb0 | (nb1 << 8u)).x;

    // iqs always divisible by 4; all 4 elements in one qs byte
    uint qs_off = base + 2u + iqs / 4u;
    uint qs_byte;
    if (binding_idx == BINDING_IDX_K) {
        qs_byte = turbo2_byte_k(qs_off);
    } else {
        qs_byte = turbo2_byte_v(qs_off);
    }

    float r0 = norm * CENTROIDS_2BIT[ qs_byte        & 0x3u];
    float r1 = norm * CENTROIDS_2BIT[(qs_byte >> 2u) & 0x3u];
    float r2 = norm * CENTROIDS_2BIT[(qs_byte >> 4u) & 0x3u];
    float r3 = norm * CENTROIDS_2BIT[(qs_byte >> 6u) & 0x3u];

    return FLOAT_TYPEV4(r0, r1, r2, r3);
}
#endif // DATA_A_TURBO2_0

#if defined(DATA_A_TURBO4_0)
// block_turbo4_0: 128 values, 66 bytes
//   bytes 0-1: float16 norm
//   bytes 2-65: qs[64]  4-bit indices, 2 per byte (low nibble first)
#define BLOCK_BYTE_SIZE 66

const float CENTROIDS_4BIT[16] = float[](
    -0.241556, -0.182907, -0.143047, -0.111065,
    -0.083317, -0.058069, -0.034311, -0.011353,
     0.011353,  0.034311,  0.058069,  0.083317,
     0.111065,  0.143047,  0.182907,  0.241556);

uint turbo4_byte_k(uint byte_abs) {
    return (k_turbo4_raw[byte_abs >> 2u] >> ((byte_abs & 3u) << 3u)) & 0xFFu;
}
uint turbo4_byte_v(uint byte_abs) {
    return (v_turbo4_raw[byte_abs >> 2u] >> ((byte_abs & 3u) << 3u)) & 0xFFu;
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    uint bi = a_offset + ib;
    uint base = bi * 66u;

    uint nb0, nb1;
    if (binding_idx == BINDING_IDX_K) {
        nb0 = turbo4_byte_k(base);
        nb1 = turbo4_byte_k(base + 1u);
    } else {
        nb0 = turbo4_byte_v(base);
        nb1 = turbo4_byte_v(base + 1u);
    }
    float norm = unpackHalf2x16(nb0 | (nb1 << 8u)).x;

    // iqs always divisible by 4; elements iqs+0,1 in one qs byte, iqs+2,3 in next
    // qs byte for element i is at block offset 2 + i/2
    uint qs0_off = base + 2u + iqs / 2u;
    uint qs1_off = qs0_off + 1u;
    uint qs0, qs1;
    if (binding_idx == BINDING_IDX_K) {
        qs0 = turbo4_byte_k(qs0_off);
        qs1 = turbo4_byte_k(qs1_off);
    } else {
        qs0 = turbo4_byte_v(qs0_off);
        qs1 = turbo4_byte_v(qs1_off);
    }

    float r0 = norm * CENTROIDS_4BIT[ qs0        & 0xFu];
    float r1 = norm * CENTROIDS_4BIT[(qs0 >> 4u) & 0xFu];
    float r2 = norm * CENTROIDS_4BIT[ qs1        & 0xFu];
    float r3 = norm * CENTROIDS_4BIT[(qs1 >> 4u) & 0xFu];

    return FLOAT_TYPEV4(r0, r1, r2, r3);
}
#endif // DATA_A_TURBO4_0

// ============================================================

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore(const in uint32_t r, const in uint32_t c, const in FLOAT_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}
