import numpy as np
import pyopencl as cl
import pyopencl.array


def main():
    dtype = np.dtype(np.int32)
    n = 1000

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    data = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            data.append(i * j)
    data = np.array(data, dtype=dtype)

    cl_data = cl.array.Array(queue, data.shape, data.dtype)
    cl_data.set(data)

    with open("palindromic.cl") as file:
        kernel = file.read()

    program = cl.Program(ctx, kernel).build()
    program.filter(queue, cl_data.shape, None, cl_data.data)
    queue.finish()

    print(np.max(cl_data.get()))


if __name__ == "__main__":
    main()
