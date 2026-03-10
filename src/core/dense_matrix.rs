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
///
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
#[derive(Debug, Clone)]
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

impl DenseMatrix {

    /// Returns the element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `row >= n_rows`
    /// - `col >= n_cols`
    ///
    /// # Complexity
    ///
    /// **O(1)**
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.n_rows);
        assert!(col < self.n_cols);

        self.data[row * self.n_cols + col]
    }

    /// Returns a mutable reference to the element at `(row, col)`.
    ///
    /// This allows in-place modification of matrix values.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `row >= n_rows`
    /// - `col >= n_cols`
    ///
    /// # Complexity
    ///
    /// **O(1)**
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        assert!(row < self.n_rows);
        assert!(col < self.n_cols);

        &mut self.data[row * self.n_cols + col]
    }

    /// Returns an immutable slice representing the specified row.
    ///
    /// This is a **zero-copy** view into the underlying data buffer.
    ///
    /// # Panics
    ///
    /// Panics if `row >= n_rows`.
    ///
    /// # Complexity
    ///
    /// **O(1)**
    #[inline]
    pub fn row(&self, row: usize) -> &[f64] {
        assert!(
            row < self.n_rows,
            "Row index out of bounds"
        );
        let start = row * self.n_cols;
        &self.data[start..start + self.n_cols]
    }

    /// Returns a mutable slice representing the specified row.
    ///
    /// This allows efficient **in-place modification** of a full row
    /// without copying data.
    ///
    /// # Panics
    ///
    /// Panics if `row >= n_rows`.
    ///
    /// # Complexity
    ///
    /// **O(1)**
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [f64] {
        assert!(row < self.n_rows);

        let start = row * self.n_cols;
        &mut self.data[start..start + self.n_cols]
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Complexity
    ///
    /// **O(1)**
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Complexity
    ///
    /// **O(1)**
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Returns the maximum value contained in the matrix.
    ///
    /// # Complexity
    ///
    /// **O(n_rows * n_cols)**
    ///
    /// # Example
    ///
    /// ```ignore
    /// let max = matrix.max();
    /// ```
    #[inline]
    pub fn max(&self) -> f64 {
        let mut max = f64::NEG_INFINITY;

        for &v in self.data.iter() {
            if v > max {
                max = v;
            }
        }

        max
    }

    /// Returns the minimum value contained in the matrix.
    ///
    /// # Complexity
    ///
    /// **O(n_rows * n_cols)**
    ///
    /// # Example
    ///
    /// ```ignore
    /// let min = matrix.min();
    /// ```
    #[inline]
    pub fn min(&self) -> f64 {
        let mut min = f64::INFINITY;

        for &v in self.data.iter() {
            if v < min {
                min = v;
            }
        }

        min
    }

    /// Creates a matrix with reserved capacity but **no initialized elements**.
    ///
    /// ⚠️ This constructor is `pub(crate)` and intended for **internal use only**.
    ///
    /// The returned matrix does **not** satisfy the invariant
    /// `data.len() == n_rows * n_cols` until all elements are inserted.
    ///
    /// Typical use cases include:
    ///
    /// - streaming data loading
    /// - CSV parsing
    /// - incremental matrix construction
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure the matrix is **fully populated**
    /// before exposing it to safe APIs.
    pub(crate) fn with_capacity(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: Vec::with_capacity(rows * cols),
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Constructs a matrix from a nested `Vec<Vec<f64>>`.
    ///
    /// Each inner vector represents a **row** of the matrix.
    ///
    /// # Panics
    ///
    /// - If the input is empty
    /// - If rows have inconsistent lengths
    ///
    /// # Complexity
    ///
    /// **O(n_rows * n_cols)**
    pub fn from_vec(vec: &[Vec<f64>]) -> DenseMatrix {
        assert!(!vec.is_empty());

        let rows = vec.len();
        let cols = vec[0].len();

        let mut data = Vec::with_capacity(rows * cols);

        for (i, row) in vec.iter().enumerate() {
            assert_eq!(row.len(), cols, "Row {} has inconsistent length", i);
            data.extend_from_slice(row);
        }

        DenseMatrix {
            data,
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Convert matrix into a nested vector representation.
    ///
    /// Each inner vector corresponds to a row of the matrix.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = matrix.to_vec();
    /// ```
    pub fn to_vec(&self) -> Vec<Vec<f64>> {

        let n_rows = self.n_rows();
        let n_cols = self.n_cols();

        let mut out = Vec::with_capacity(n_rows);

        for i in 0..n_rows {

            let start = i * n_cols;
            let end = start + n_cols;

            out.push(self.data[start..end].to_vec());
        }
        out
    }

    /// Returns an immutable view of the underlying contiguous data buffer.
    ///
    /// The returned slice contains all elements in **row-major order**.
    ///
    /// # Layout
    ///
    /// The slice is organized as:
    ///
    /// ```text
    /// [ r0c0, r0c1, ..., r0cN,
    ///   r1c0, r1c1, ..., r1cN,
    ///   ...
    /// ]
    /// ```
    ///
    /// This method is useful for:
    ///
    /// - High-performance iteration
    /// - Bulk operations on matrix elements
    /// - Interfacing with numeric libraries
    ///
    /// # Complexity
    ///
    /// **O(1)** — no allocation or copying occurs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slice = matrix.as_slice();
    /// for v in slice {
    ///     println!("{}", v);
    /// }
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable view of the underlying contiguous data buffer.
    ///
    /// This allows **in-place modification** of all matrix elements
    /// without copying the data.
    ///
    /// # Layout
    ///
    /// The slice is stored in **row-major order**.
    ///
    /// # Safety
    ///
    /// Modifying values through this slice will **not break the internal
    /// shape invariants** (`n_rows`, `n_cols`) because the length of the
    /// buffer cannot change.
    ///
    /// # Complexity
    ///
    /// **O(1)** — no allocation or copying occurs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slice = matrix.as_slice_mut();
    ///
    /// for v in slice {
    ///     *v += 1.0;
    /// }
    /// ```
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Creates a matrix filled with **zeros**.
    ///
    /// # Complexity
    ///
    /// **O(rows * cols)**
    pub fn zeros(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: vec![0.0; rows * cols],
            n_rows: rows,
            n_cols: cols,
        }
    }

    /// Creates a matrix filled with **ones**.
    ///
    /// # Complexity
    ///
    /// **O(rows * cols)**
    pub fn ones(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            data: vec![1.0; rows * cols],
            n_rows: rows,
            n_cols: cols,
        }
    }

}
