import pyopencl as cl
import pyopencl.array
import numpy as np
from itertools import islice


def gen_data(n):
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = n - a - b
            yield [a, b, c]


def main():
    dtype = np.dtype(np.uint32)
    data = np.array(list(gen_data(1000))).astype(dtype)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    cl_data = cl.array.Array(queue, data.shape, dtype)
    cl_data.set(data)
    cl_result = cl.array.zeros(queue, data.shape[0], dtype)

    with open("kernel.cl") as file:
        kernel = file.read()
    program = cl.Program(ctx, kernel).build()

    program.process(queue, cl_data.shape, None, cl_data.data, cl_result.data)
    queue.finish()

    result = cl_result.get()
    result = result[result != 0]
    print(result[0])


if __name__ == "__main__":
    main()
