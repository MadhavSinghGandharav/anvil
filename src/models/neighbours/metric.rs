use crate::models::neighbours::DistanceMetric;
use ndarray::{ArrayView1, Zip};

/// Euclidean (L2) distance metric.
///
/// Computes:
///
/// √(Σ (aᵢ - bᵢ)²)
pub struct Euclidean;

/// Manhattan (L1) distance metric.
///
/// Computes:
///
/// Σ |aᵢ - bᵢ|
pub struct Manhattan;

impl DistanceMetric for Euclidean{
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        
        let sum = Zip::from(a)
            .and(b)
            .fold(0.0, |acc,&a,&b| {
                let diff = a-b;
                acc + diff*diff
            });
        sum.sqrt()
    }
}

impl DistanceMetric for Manhattan{
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        
        Zip::from(a)
            .and(b)
            .fold(0.0, |acc,&a,&b| {
                let diff = a-b;
                acc + diff.abs()
            })        
    }
}
