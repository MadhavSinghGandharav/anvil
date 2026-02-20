use crate::neighbours::DistanceMetric;

/// Euclidean distance metric.
///
/// Computes the L2 distance between two vectors:
///
/// √(Σ (aᵢ - bᵢ)²)
///
/// Commonly used in standard KNN implementations.
pub struct Euclidean;

/// Manhattan distance metric.
///
/// Computes the L1 distance between two vectors:
///
/// Σ |aᵢ - bᵢ|
///
/// Often more robust to outliers compared to Euclidean distance.
pub struct Manhattan;

impl DistanceMetric for Euclidean {

    /// Computes Euclidean (L2) distance between two vectors.
    ///
    /// # Panics
    /// Panics if the vectors have different lengths.
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let mut sum = 0.0;

        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

impl DistanceMetric for Manhattan {

    /// Computes Manhattan (L1) distance between two vectors.
    ///
    /// # Panics
    /// Panics if the vectors have different lengths.
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let mut sum = 0.0;

        for i in 0..a.len() {
            sum += (a[i] - b[i]).abs();
        }

        sum
    }
}
