import numpy as np
import pyopencl as cl
import pyopencl.array


def main():
    dtype = np.dtype(np.uint32)
    n = 1_000_000

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    cl_data = cl.array.arange(queue, 1, n + 1, dtype=dtype)

    with open("primes.cl") as file:
        kernel = file.read()
    program = cl.Program(ctx, kernel).build()

    e = program.primes(queue, cl_data.shape, None, cl_data.data)
    e.wait()
    queue.finish()

    result = cl_data.get()
    result = result[result != 0]
    print(result[10_000])


if __name__ == "__main__":
    main()
