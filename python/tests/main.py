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
    print("error = ", np.linalg.norm(mat - mat0))

eval_mateval(np.float32)
eval_mateval(np.float64)
eval_mateval(np.int8)
eval_mateval(np.int16)
eval_mateval(np.int32)
eval_mateval(np.int64)
eval_mateval(np.uint8)
eval_mateval(np.uint16)
eval_mateval(np.uint32)
eval_mateval(np.uint64)
