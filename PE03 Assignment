import numpy as np

# Create matrix A as specified
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Create matrix B as a 3 x3 identity matrix
B = np.eye(3)

print("Matrix A:\n", A)
print("Matrix B (Identity Matrix):\n", B)

# Compute the matrix product of A and B
C = np.dot(A, B)
print("Matrix C (A * B):\n", C)

# Add 5 to each element of matrix C
C = C + 5
print("Matrix C after adding 5 to each element:\n", C)

# Compute the square root of each element in the modified matrix C
C = np.sqrt(C)
print("Matrix C after taking the square root of each element:\n", C)

# Compute the sum of each row in matrix C
row_sums = np.sum(C, axis=1)
print("Sum of each row in matrix C:", row_sums)

# Compute the sum of each column in matrix C
column_sums = np.sum(C, axis=0)
print("Sum of each column in matrix C:", column_sums)

# Compute the sum of each row in matrix C
row_sums = np.sum(C, axis=1)
print("Sum of each row in matrix C:", row_sums)

# Compute the sum of each column in matrix C
column_sums = np.sum(C, axis=0)
print("Sum of each column in matrix C:", column_sums)

# Transpose the matrix C
C_transposed = C.T
print("Transposed Matrix C:\n", C_transposed)

# Flatten the transposed matrix into a 1D array
C_flattened = C_transposed.flatten()
print("Flattened Transposed Matrix C:", C_flattened)

# Compute the determinant of matrix A
det_A = np.linalg.det(A)
print("Determinant of Matrix A:", det_A)

# Compute the eigenvalues and eigenvectors of matrix A
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of Matrix A:", eigenvalues)
print("Eigenvectors of Matrix A:\n", eigenvectors)

# Function to normalize a matrix
def normalize_matrix(mat):
    min_val = np.min(mat)
    max_val = np.max(mat)
    normalized_mat = (mat - min_val) / (max_val - min_val)
    return normalized_mat

# Apply normalization to matrix C
normalized_C = normalize_matrix(C)
print("Normalized Matrix C:\n", normalized_C)
