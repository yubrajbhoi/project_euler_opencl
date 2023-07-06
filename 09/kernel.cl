__kernel void process(__global uint* data, __global uint* result) {
    int gid = get_global_id(0);
    uint a = data[gid * 3 + 0];
    uint b = data[gid * 3 + 1];
    uint c = data[gid * 3 + 2];

    if (pown((float)a, 2) + pown((float)b, 2) == pown((float)c, 2)) {
        result[gid] = a * b * c;
    }
}
