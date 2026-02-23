use crate::core::DenseMatrix;
use crate::neighbours::{DistanceMetric,NeighbourSearch};

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
pub struct BruteForce{
    features: Option<DenseMatrix>,
}

impl BruteForce{
    pub fn new() -> Self{
        Self{
            features: None
        }
    }
}
impl <M: DistanceMetric> NeighbourSearch<M> for BruteForce {

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
    ///
    fn fit(&mut self, features: DenseMatrix) {
        self.features = Some(features);
    }

    fn neighbours(
        &self,
        query: &[f64],
        k: usize,
        metric: &M,
    ) -> Vec<(f64, usize)> {
        
        let features = self.features.as_ref().expect("model not fitted");

        assert_eq!(features.n_cols(),query.len());
        
        let mut distances: Vec<(f64, usize)> = (0..features.n_rows())
            .map(|row| {
                let row_slice = features.row(row);
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


