use ndarray::{Array2, ArrayView1};
use crate::{models::neighbours::{DistanceMetric, NeighbourSearch}, neighbours::metric::Euclidean};

/// Brute-force nearest neighbour search.
///
/// Computes the distance between the query point and every row
/// in the training matrix, then selects the `k` smallest distances
/// using partial selection.
///
/// # Complexity
///
/// ```text
/// Distance computation : O(n × features)
/// Selection (Quickselect): O(n) average
/// ```
pub struct BruteForce<M: DistanceMetric> {

    /// Training feature matrix of shape `(n_samples, n_features)`
    features: Option<Array2<f64>>,

    /// Distance metric used for neighbour search
    metric: M,
}

impl BruteForce<Euclidean> {
    pub fn new() -> Self {
        Self {
            features: None,
            metric: Euclidean,
        }
    }
}

impl<M: DistanceMetric> BruteForce<M> {
    pub fn with_metric(metric: M) -> Self {
        Self {
            features: None,
            metric,
        }
    }
}
impl<M: DistanceMetric> NeighbourSearch for BruteForce<M> {

    /// Stores the training matrix and metric for future queries
    fn build(&mut self, data: Array2<f64>) {
        self.features = Some(data);
    }

    /// Returns the `k` nearest neighbours for a given query point
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - query point dimension does not match training data
    /// - `k` is zero
    fn query(&self, point: ArrayView1<f64>, k: usize) -> Vec<(usize, f64)> {
        
        let features = self.features.as_ref().expect("Apply build first");
        assert!(k > 0, "k must be greater than 0");

        assert_eq!(
            features.ncols(),
            point.len(),
            "Query point dimension mismatch"
        );

        // compute distance from query point to every training row
        let mut distances: Vec<(usize, f64)> = features
            .outer_iter()
            .enumerate()
            .map(|(i, row)| (i, self.metric.distance(row, point)))
            .collect();

        // ensure k does not exceed dataset size
        let kth = k.min(distances.len());

        // Quickselect: places the kth smallest element in its correct
        // position — elements before it are smaller, after it are larger
        distances.select_nth_unstable_by(kth - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });

        // keep only the k smallest elements (not guaranteed to be sorted)
        distances.truncate(kth);
        distances
    }
}
