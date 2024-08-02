import matfile
import numpy as np


def eval_mateval(dtype):
    print("## ", dtype)
    mat = np.random.rand(3, 2).astype(dtype)
    matfile.save_dense(mat, "test.matrix")
    mat0 = matfile.load_dense("test.matrix")
    print("saved shape = ", mat.shape)
    print("saved dtype = ", mat.dtype)
    print("loaded shape = ", mat0.shape)
    print("loaded dtype = ", mat0.dtype)
    error = np.linalg.norm(mat - mat0)
    print("error = ", error)
    if error == 0:
        return 0
    return 1

num_errors = 0
num_errors += eval_mateval(np.float32)
num_errors += eval_mateval(np.float64)
num_errors += eval_mateval(np.int8)
num_errors += eval_mateval(np.int16)
num_errors += eval_mateval(np.int32)
num_errors += eval_mateval(np.int64)
num_errors += eval_mateval(np.uint8)
num_errors += eval_mateval(np.uint16)
num_errors += eval_mateval(np.uint32)
num_errors += eval_mateval(np.uint64)

if __name__ == "__main__":
    print(f"Num errors = {num_errors}")
    if num_errors == 0:
        exit(0)
    else:
        exit(1)
