/// A dense 1-dimensional vector backed by a contiguous `Vec<f64>`.
///
/// This type represents a mathematical vector stored in
/// contiguous memory for cache-efficient access.
///
/// # Usage
///
/// `DenseVector` is commonly used for:
/// - Target values (`y`)
/// - Model weights (`w`)
/// - Gradients
/// - Intermediate buffers in optimization algorithms
///
/// # Memory Layout
///
/// Elements are stored in row-major contiguous order:
///
/// `[v0, v1, v2, ..., vn]`
///
/// # Invariants
///
/// - The internal buffer length defines the vector dimension.
/// - Length must match expected dimensionality in model operations.
///
/// # Performance
///
/// - Element access is **O(1)**.
/// - `as_slice()` and `as_mut_slice()` are zero-cost operations.
/// - Designed to work efficiently with slice-based math kernels.
#[allow(dead_code)]
pub struct DenseVector {
    /// Contiguous data buffer.
    ///
    /// The length of this buffer defines the vector dimension.
    data: Vec<f64>,
}

impl DenseVector {
    /// Returns the element at index `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len()`.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn get(&self, idx: usize) -> f64 {
        assert!(
            idx < self.data.len(),
            "Index {} out of bounds for vector of length {}",
            idx,
            self.data.len()
        );
        self.data[idx]
    }

    /// Returns the length (dimension) of the vector.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Creates a `DenseVector` from an existing `Vec<f64>`.
    ///
    /// This constructor does not copy the data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let v = DenseVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn from_vec(arr: Vec<f64>) -> Self {
        Self { data: arr }
    }

    /// Returns an immutable slice of the underlying data.
    ///
    /// This is a zero-cost view into the internal buffer.
    ///
    /// Useful for:
    /// - Dot products
    /// - Optimizer updates
    /// - Passing data to math kernels
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    /// Returns a mutable slice of the underlying data.
    ///
    /// This enables in-place modification of vector elements.
    ///
    /// Commonly used by optimizers to update parameters.
    ///
    /// # Safety
    ///
    /// The caller must ensure dimension consistency when
    /// performing mathematical operations.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }

    /// Creates a `DenseVector` filled with zeros.
    ///
    /// # Panics
    ///
    /// Panics if `n` causes memory allocation failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let v = DenseVector::zeros(5);
    /// ```
    pub fn zeros(n: usize) -> Self {
        DenseVector {
            data: vec![0.0f64; n],
        }
    }
}

use std::ops::{Index, IndexMut};

/// Enables immutable indexing using `vector[i]`.
///
/// # Panics
///
/// Panics if `index >= self.len()`.
///
/// # Example
///
/// ```ignore
/// let v = DenseVector::from_vec(vec![1.0, 2.0]);
/// let x = v[0];
/// ```
impl Index<usize> for DenseVector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

/// Enables mutable indexing using `vector[i] = value`.
///
/// # Panics
///
/// Panics if `index >= self.len()`.
///
/// # Example
///
/// ```ignore
/// let mut v = DenseVector::zeros(3);
/// v[1] = 5.0;
/// ```
impl IndexMut<usize> for DenseVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
