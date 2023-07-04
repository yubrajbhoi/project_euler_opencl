__kernel void process(__global ulong* data, __global ulong* result, int size) {
    int gid = get_global_id(0);
    ulong mul_result = 1;

    for (int i = 0; i < size; i++) {
        mul_result *= data[gid * size + i];
    }
    result[gid] = mul_result;
}
