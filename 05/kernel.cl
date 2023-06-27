bool is_num(ulong n) {
    ulong i;
    for (i = 1; i <= 20; i++) {
        if (n % i != 0)
            return false;
    }
    return true;
}

__kernel void filter(__global ulong *data) {
    int i = get_global_id(0);

    if (!is_num(data[i])) {
        data[i] = 0;
    }
}
