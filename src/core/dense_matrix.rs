
/// A dense 2-dimensional matrix stored in **row-major** order.
///
/// Internally, elements are stored in a contiguous `Vec<f64>` buffer.
/// The element at position `(row, col)` is located at:
///
/// `row * n_cols + col`
///
/// # Memory Layout
///
/// Data is stored row-by-row:
///
/// ```text
/// [ r0c0, r0c1, ..., r0cN,
///   r1c0, r1c1, ..., r1cN,
///   ...
/// ]
/// ```
///
/// This layout provides:
/// - Cache-friendly row-wise iteration
/// - Efficient per-sample access (useful for SGD)
/// - Compatibility with BLAS when layout is specified correctly
///
/// # Invariants
///
/// - `data.len() == n_rows * n_cols`
///
/// All public constructors (`from_vec`, `zeros`, `ones`)
/// guarantee that this invariant holds.
///
/// # Performance
///
/// - Element access: **O(1)**
/// - Row slicing: **O(1)** (zero-copy view)
/// - Construction: **O(n_rows * n_cols)**
///
/// # Panics
///
/// - Indexing methods panic on out-of-bounds access.
/// - `from_vec` panics if rows have inconsistent lengths.
#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct DenseMatrix {
    /// Contiguous row-major data buffer.
    ///
    /// Length must always equal `n_rows * n_cols`.
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
    ///
    /// Panics if:
    /// - `row >= n_rows`
    /// - `col >= n_cols`
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.n_rows,
            "Row index {} out of bounds for matrix with {} rows",
            row,
            self.n_rows
        );
        assert!(
            col < self.n_cols,
            "Column index {} out of bounds for matrix with {} columns",
            col,
            self.n_cols
        );

        self.data[row * self.n_cols + col]
    }

    /// Returns an immutable slice representing the specified row.
    ///
    /// This is a zero-copy, zero-allocation view into the
    /// underlying contiguous buffer.
    ///
    /// # Panics
    ///
    /// Panics if `row >= n_rows`.
    ///
    /// # Complexity
    ///
    /// O(1)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let row = matrix.row(0);
    /// let first_value = row[0];
    /// ```
    #[inline]
    pub fn row(&self, row: usize) -> &[f64] {
        assert!(
            row < self.n_rows,
            "Row index {} out of bounds for matrix with {} rows",
            row,
            self.n_rows
        );

        let start = row * self.n_cols;
        &self.data[start..start + self.n_cols]
    }

    /// Returns the number of rows.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Returns the number of columns.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Creates a matrix with reserved capacity but **no initialized elements**.
    ///
    /// ⚠️ This constructor is `pub(crate)` and intended for internal use
    /// (e.g., CSV parsing or streaming construction).
    ///
    /// The returned matrix does **not** satisfy the invariant
    /// `data.len() == n_rows * n_cols` until it is fully populated.
    ///
    /// # Warning
    ///
    /// This function should only be used in controlled internal contexts
    /// where the caller guarantees proper filling of all elements.
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
    ///
    /// - If input is empty.
    /// - If row lengths are inconsistent.
    ///
    /// # Complexity
    ///
    /// O(n_rows * n_cols)
    pub fn from_vec(vec: &[Vec<f64>]) -> DenseMatrix {
        assert!(!vec.is_empty(), "Input matrix cannot be empty");

        let rows = vec.len();
        let cols = vec[0].len();

        let mut data = Vec::with_capacity(rows * cols);

        for (i, row) in vec.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "Row {} has length {}, expected {}",
                i,
                row.len(),
                cols
            );
            data.extend_from_slice(row);
        }

        DenseMatrix {
            data,
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Creates a matrix filled with zeros.
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols` causes allocation failure.
    ///
    /// # Complexity
    ///
    /// O(rows * cols)
    pub fn zeros(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: vec![0.0; rows * cols],
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Creates a matrix filled with ones.
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols` causes allocation failure.
    ///
    /// # Complexity
    ///
    /// O(rows * cols)
    pub fn ones(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: vec![1.0; rows * cols],
            n_rows: rows,
            n_cols: cols,
        }
    }
}
