
use crate::core::{DenseMatrix, DenseVector};

/// Performs matrix–vector multiplication.
///
/// Computes:
///
/// `result = matrix × vector`
///
/// where:
/// - `matrix` has shape `(n_samples, n_features)`
/// - `vector` has length `n_features`
///
/// The returned `DenseVector` has length `n_samples`.
///
/// # Panics
///
/// Panics if:
/// - `matrix.n_cols() != vector.len()`
///
/// # Complexity
///
/// O(n_samples * n_features)
///
/// # Performance Notes
///
/// - Uses row-wise access, which is cache-friendly
///   due to row-major memory layout.
/// - Allocates a new `DenseVector` for the result.
pub fn mat_vec_mul(matrix: &DenseMatrix, vector: &[f64]) -> DenseVector {
    let n_features = matrix.n_cols();
    let n_samples = matrix.n_rows();

    assert_eq!(n_features, vector.len());

    let mut result = DenseVector::zeros(n_samples);

    for row in 0..n_samples {
        let row_slice = matrix.row(row);
        let mut sum = 0.0;

        for col in 0..n_features {
            sum += row_slice[col] * vector[col];
        }

        result[row] = sum;
    }

    result
}

/// Computes the dot product of two vectors.
///
/// Computes:
///
/// `v1 · v2 = Σ (v1[i] * v2[i])`
///
/// # Panics
///
/// Panics if:
/// - `v1.len() != v2.len()`
///
/// # Complexity
///
/// O(n)
///
/// # Performance Notes
///
/// - Uses simple scalar accumulation.
/// - Suitable for small to medium-sized vectors.
/// - Can later be optimized with SIMD if needed.
pub fn dot(v1: &[f64], v2: &[f64]) -> f64 {
    assert_eq!(v1.len(), v2.len());

    let mut result: f64 = 0.0;

    for i in 0..v1.len() {
        result += v1[i] * v2[i];
    }

    result
}
