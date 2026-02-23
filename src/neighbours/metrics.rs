use crate::neighbours::DistanceMetric;

/// Euclidean (L2) distance metric.
///
/// Computes:
///
/// √(Σ (aᵢ - bᵢ)²)
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

/// Manhattan (L1) distance metric.
///
/// Computes:
///
/// Σ |aᵢ - bᵢ|
#[derive(Clone, Copy, Debug, Default)]
pub struct Manhattan;

impl DistanceMetric for Euclidean {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {

        let mut sum = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

impl DistanceMetric for Manhattan {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {

        let mut sum = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += (x - y).abs();
        }

        sum
    }
}
