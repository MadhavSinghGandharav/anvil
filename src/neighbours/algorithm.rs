use crate::core::DenseMatrix;
use crate::neighbours::DistanceMetric;

/// Brute-force neighbor search implementation.
///
/// This strategy computes the distance between the query point
/// and every row in the training matrix, then selects the
/// `k` smallest distances using partial selection.
///
/// Time Complexity:
/// - Distance computation: O(n)
/// - Selection (Quickselect): O(n) average
///
/// Overall average complexity: O(n)
pub(crate) struct BruteForce;

impl BruteForce {

    /// Returns the `k` nearest neighbors for a given query point.
    ///
    /// # Arguments
    /// * `train`  - Training feature matrix
    /// * `query`  - Single query feature slice
    /// * `k`      - Number of neighbors to retrieve
    /// * `metric` - Distance metric implementation
    ///
    /// # Returns
    /// A vector of `(distance, row_index)` pairs for the
    /// `k` closest rows in the training data.
    ///
    /// The returned neighbors are not guaranteed to be sorted.
    pub fn neighbours<M: DistanceMetric>(
        train: &DenseMatrix,
        query: &[f64],
        k: usize,
        metric: &M,
    ) -> Vec<(f64, usize)> {

        let mut distances: Vec<(f64, usize)> = (0..train.n_rows())
            .map(|row| {
                let row_slice = train.row(row);
                let dist = metric.distance(row_slice, query);
                (dist, row)
            })
            .collect();

        // Ensure k does not exceed dataset size
        let kth = k.min(distances.len());

        // Quickselect partitioning: places the k-th smallest element
        // in its correct position and partitions the vector.
        distances.select_nth_unstable_by(kth - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        // Keep only the k smallest elements
        distances.truncate(kth);

        distances
    }
}

