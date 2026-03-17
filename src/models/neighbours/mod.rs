mod brute_force;
pub mod metric;
mod kdtree;
mod knn_clf;
mod knn_reg;

pub use knn_clf::KNNClassifier;
pub use kdtree::KDTree;
pub use brute_force::BruteForce;
pub use knn_reg::KNNRegressor;

use ndarray::{ArrayView1,Array2};

pub trait DistanceMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64;
}

pub trait NeighbourSearch{
    fn build(&mut self, data: Array2<f64>);
    fn query(&self, point: ArrayView1<f64>, k: usize) -> Vec<(usize, f64)>;
}

#[derive(Clone, Copy, Debug)]
pub enum Weight {
    /// All neighbors contribute equally.
    Uniform,

    /// Each neighbor contributes weight = 1 / distance.
    ///
    /// If distance is zero, weight is treated as 1.0
    /// to avoid division by zero.
    Distance,
}




