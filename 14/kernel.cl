/*
 * Count the sequence for `n` using this algorithm
 * n -> n/2 (if n is even)
 * n -> 3n + 1 (if n is odd)
 *
 * Stop when `n` is one.
 */
uint count_seq(uint n) {
    uint count = 1;
    while (n != 1) {
        if (n % 2 == 0)
            n = n / 2;
        else
            n = (3 * n) + 1;
        count++;
    }
    return count;
}

__kernel void process(__global uint* a) {
    int i = get_global_id(0);
    a[i] = count_seq(a[i]);
}
