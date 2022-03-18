import matfile
import numpy as np

mat = np.random.rand(2, 3).astype('f')

matfile.save_dense(mat, "test.matrix")

mat0 = np.array([])
if matfile.get_fp_bit == 32:
    mat0 = matfile.load_dense_fp32("test.matrix")
else:
    mat0 = matfile.load_dense_fp64("test.matrix")

print(mat.shape)
print(mat)
print(mat0.shape)
print(mat0)
print("error = ", np.linalg.norm(mat - mat0))
