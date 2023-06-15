import numpy as np
import pyopencl as cl
import pyopencl.array


def main() -> None:
    dtype = "int32"
    n = 2_000_000

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    x = np.arange(1, n + 1, dtype=dtype)
    clx = cl.array.Array(queue, x.shape, dtype)
    clx.set(x)

    with open("primes.cl") as file:
        kernel = file.read()
    prg = cl.Program(
        ctx,
        kernel,
    ).build()

    prg.primes(queue, clx.shape, None, clx.data)
    queue.finish()

    data = clx.get()
    primes = [i for i in data if i != 0]
    print(sum(primes))


if __name__ == "__main__":
    main()
