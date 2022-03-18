import matfile
import numpy as np

mat = np.random.rand(2, 3)

matfile.save_dense_fp64(mat, "test.matrix")

mat0 = np.array([])
fpbit = matfile.get_fp_bit("test.matrix")
print(fpbit)
if fpbit == 32:
    print("Load FP32 matrix")
    mat0 = matfile.load_dense_fp32("test.matrix")
else:
    print("Load FP64 matrix")
    mat0 = matfile.load_dense_fp64("test.matrix")

print("saved shape = ", mat.shape)
print("loaded shape = ", mat0.shape)
print("error = ", np.linalg.norm(mat - mat0))
