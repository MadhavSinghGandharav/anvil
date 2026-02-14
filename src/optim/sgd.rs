use crate::optim::Optimizer;

pub struct SGD{
    lr: f64
}
impl SGD{
    pub fn new(lr: f64) -> Self{
        Self{
            lr
        }
    }
}

impl Optimizer for SGD{
    fn step(&mut self, weights: &mut [f64], gradient: &[f64]) {
       for i in 0..weights.len() {
          weights[i] -= self.lr * gradient[i]; 
       }
    }
}
