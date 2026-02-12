/// A dense 2-dimensional matrix stored in **row-major** order.
///
/// Internally, elements are stored in a contiguous `Vec<f64>` buffer.
/// The element at position `(row, col)` is located at:
///
/// `row * n_cols + col`
///
/// This layout provides:
/// - Cache-friendly memory access
/// - Efficient row-wise iteration (useful for SGD and inference)
/// - Compatibility with BLAS (when layout is specified correctly)
///
/// # Invariants
/// - `data.len() == n_rows * n_cols`
///
/// All public constructors (`from_vec`, `zeros`, `ones`)
/// guarantee that this invariant holds.
///
/// # Panics
/// - Indexing methods panic on out-of-bounds access.
/// - `from_vec` panics if rows have inconsistent lengths.
#[allow(dead_code)]
pub struct DenseMatrix {
    /// Contiguous row-major data buffer.
    data: Vec<f64>,

    /// Number of rows in the matrix.
    n_rows: usize,

    /// Number of columns in the matrix.
    n_cols: usize,
}

#[allow(dead_code)]
impl DenseMatrix {
    /// Returns the element at `(row, col)`.
    ///
    /// # Panics
    /// Panics if `row` or `col` are out of bounds.
    ///
    /// # Complexity
    /// O(1)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.n_cols + col]
    }

    /// Returns a slice representing the specified row.
    ///
    /// This is a zero-cost operation due to row-major layout.
    ///
    /// # Panics
    /// Panics if `row` is out of bounds.
    ///
    /// # Complexity
    /// O(1)
    #[inline]
    pub fn row(&self, row: usize) -> &[f64] {
        let start = row * self.n_cols;
        &self.data[start..start + self.n_cols]
    }

    /// Creates a matrix with reserved capacity but **no initialized elements**.
    ///
    /// ⚠️ This constructor is `pub(crate)` and intended for internal use
    /// (e.g., CSV parsing or streaming construction).
    ///
    /// The returned matrix does **not** satisfy the invariant
    /// `data.len() == n_rows * n_cols` until it is fully filled.
    pub(crate) fn with_capacity(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: Vec::with_capacity(rows * cols),
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Constructs a matrix from a nested `Vec<Vec<f64>>`.
    ///
    /// All inner vectors must have equal length.
    ///
    /// # Panics
    /// - If input is empty.
    /// - If row lengths are inconsistent.
    pub fn from_vec(vec: &[Vec<f64>]) -> DenseMatrix {
        assert!(!vec.is_empty(), "Input matrix cannot be empty");

        let rows = vec.len();
        let cols = vec[0].len();

        let mut data = Vec::with_capacity(rows * cols);

        for row in vec {
            assert_eq!(row.len(), cols);
            data.extend_from_slice(row);
        }

        DenseMatrix {
            data,
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Creates a matrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: vec![0.0; rows * cols],
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Creates a matrix filled with ones.
    pub fn ones(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: vec![1.0; rows * cols],
            n_rows: rows,
            n_cols: cols,
        }
    }
}

/// A dense 1-dimensional vector backed by a contiguous `Vec<f64>`.
///
/// Used for:
/// - Target values (`y`)
/// - Model weights (`w`)
/// - Intermediate buffers
///
/// # Invariants
/// - Length must match expected dimensionality in model operations.
#[allow(dead_code)]
pub struct DenseVector {
    /// Contiguous data buffer.
    data: Vec<f64>,
}
