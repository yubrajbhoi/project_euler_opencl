import numpy as np
import pyopencl as cl
import pyopencl.array


def main() -> None:
    dtype = np.dtype(np.int32)
    n = 2_000_000

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    clx = cl.array.arange(queue, 1, n + 1, dtype)

    with open("primes.cl") as file:
        kernel = file.read()
    prg = cl.Program(
        ctx,
        kernel,
    ).build()

    prg.primes(queue, clx.shape, None, clx.data)
    queue.finish()
    print(sum(clx.get()))


if __name__ == "__main__":
    main()
