mod knn;
mod metrics;
pub(crate) mod algorithm;

/// Trait defining a distance metric.
///
/// Any distance metric used in KNN must implement this trait.
/// The `distance` function computes the distance between two feature vectors.
pub trait DistanceMetric {
    /// Computes distance between two feature slices.
    ///
    /// # Arguments
    /// * `a` - First feature vector
    /// * `b` - Second feature vector
    ///
    /// # Returns
    /// A `f64` representing the distance.
    fn distance(&self, a: &[f64], b: &[f64]) -> f64;
}

/// Neighbor search strategy used by KNN.
///
/// Currently supported:
/// - `Brute` — Brute-force search over all training samples.
///
/// Future extensions may include KD-Tree or Ball-Tree.
pub enum Algorithm {
    /// Brute-force neighbor search.
    Brute,
}

/// Weighting strategy applied during neighbor aggregation.
///
/// - `Uniform` — All neighbors contribute equally.
/// - `Distance` — Neighbors are weighted inversely proportional to distance.
pub enum Weight {
    /// Equal weight for all neighbors.
    Uniform,

    /// Weight = 1 / distance.
    Distance,
}

/// Re-export of Euclidean distance metric.
pub use metrics::Euclidean;

/// Re-export of Manhattan distance metric.
pub use metrics::Manhattan;

/// Re-export of KNN Regressor model.
pub use knn::KNNRegressor;
