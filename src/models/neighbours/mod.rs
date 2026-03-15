mod brute_force;
mod metric;
mod kdtree;

use ndarray::{ArrayView1,Array2};

pub trait DistanceMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64;
}

pub trait NeighbourSearch <M: DistanceMetric>{
    fn build(data: Array2<f64>, metric: M) -> Self;
    fn query(&self, point: ArrayView1<f64>, k: usize) -> Vec<(usize, f64)>;
}




