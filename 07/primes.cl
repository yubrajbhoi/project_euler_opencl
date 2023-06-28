bool is_prime(uint n) {
    uint i;
    if (n < 2) return false;

    float sqrt_n = sqrt((float)n);
    for (i = 2; i <= sqrt_n; i++) {
        if (n % i == 0)
            return false;
    }

    return true;
}

__kernel void primes(__global uint* a) {
    int i = get_global_id(0);
    if (!is_prime(a[i])) a[i] = 0;
}
