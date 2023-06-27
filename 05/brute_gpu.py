import numpy as np
import pyopencl as cl
import pyopencl.array


def main():
    dtype = np.dtype(np.uint64)
    # Increment for each kernel (per loop)
    n = 10_000_000

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    with open("kernel.cl") as file:
        kernel = file.read()
    program = cl.Program(ctx, kernel).build()

    start, end = 1, n
    while True:
        cl_data = cl.array.arange(queue, start, end, dtype=dtype)
        e = program.filter(queue, cl_data.shape, None, cl_data.data)
        e.wait()

        result = cl_data.get()
        result = result[result != 0]
        print(f"Range {start} to {end} done.")

        if result.size > 0:
            break
        start, end = end, end + n

    queue.finish()
    print(result[0])


if __name__ == "__main__":
    main()
