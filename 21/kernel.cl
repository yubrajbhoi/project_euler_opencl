/*
 * Sum of all the proper divisors of `n`.
 */
uint sum_div(uint n) {
    uint count = 0, i;
    for (i = 1; i < n; i++) {
        if (n % i == 0) {
            count = count + i;
        }
    }
    return count;
}

/*
 * Calculate if `n` is an amicable number.
 */
bool apair(uint n) {
    uint r = sum_div(n);
    if (r != n)
        return (n == sum_div(r));
    return false;
}

__kernel void process(__global uint* a) {
    int i = get_global_id(0);
    if (!apair(a[i])) {
        a[i] = 0;
    }
}
