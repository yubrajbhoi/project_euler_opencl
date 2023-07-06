/*
 * Take an array of 2D data, with shape of (n, 3). The result array is 1D of length `n`.
 *
 * a, b and c are elements of each inner array. Store the product of these elelents (a * b * c) where (a ^ 2 + b ^ 2) == c ^ 2.
 */
__kernel void process(__global uint* data, __global uint* result) {
    int gid = get_global_id(0);
    uint a = data[gid * 3 + 0];
    uint b = data[gid * 3 + 1];
    uint c = data[gid * 3 + 2];

    if (pown((float)a, 2) + pown((float)b, 2) == pown((float)c, 2)) {
        result[gid] = a * b * c;
    }
}
