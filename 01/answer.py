import numpy as np
import pyopencl as cl
import pyopencl.array


KERNEL = """
__kernel void filter(__global int *x) {
    int gid = get_global_id(0);
    int i = x[gid];
    if (i % 3 != 0 && i % 5 != 0) {
        x[gid] = 0;
    }
}
"""


def main():
    dtype = np.dtype(np.int32)
    n = 1000

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    clx = cl.array.arange(queue, 1, n + 1, dtype)

    prg = cl.Program(
        ctx,
        KERNEL,
    ).build()

    prg.filter(queue, clx.shape, None, clx.data)
    queue.finish()
    print(sum(clx))


if __name__ == "__main__":
    main()
