bool palindromic(int n) {
    int reverse = 0, temp = n, rem;
    while (temp != 0) {
        rem = temp % 10;
        reverse = reverse * 10 + rem;
        temp /= 10;
    }

    return (n == reverse);
}


__kernel void filter(__global int *data) {
    int i = get_global_id(0);

    if (!palindromic(data[i])) {
        data[i] = 0;
    }
}
