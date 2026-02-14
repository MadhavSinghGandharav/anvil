
mod sgd;

pub trait Optimizer{
    fn step(&mut self, weights: &mut [f64], gradient: &[f64]);
}
