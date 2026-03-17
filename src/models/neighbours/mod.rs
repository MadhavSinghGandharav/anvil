mod brute_force;
mod metric;
mod kdtree;

use ndarray::{ArrayView1,Array2};

pub trait DistanceMetric {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64;
}

pub trait NeighbourSearch{
    fn build(&mut self, data: Array2<f64>);
    fn query(&self, point: ArrayView1<f64>, k: usize) -> Vec<(usize, f64)>;
}




