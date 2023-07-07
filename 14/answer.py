import pyopencl as cl
import pyopencl.array
import numpy as np


def main():
    n = 1_000_000
    dtype = np.dtype(np.uint32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    cl_data = cl.array.arange(queue, 1, n, dtype)

    with open("kernel.cl") as file:
        kernel = file.read()
    program = cl.Program(ctx, kernel).build()

    program.process(queue, cl_data.shape, None, cl_data.data)
    queue.finish()

    data = cl_data.get()
    # Get the index of the maximum value, and add one (because of 0 based index)
    max_seq = np.max(data)
    result = np.where(data == max_seq)[0][0] + 1
    print(f"Answer: {result}, Seq: {max_seq}")


if __name__ == "__main__":
    main()
