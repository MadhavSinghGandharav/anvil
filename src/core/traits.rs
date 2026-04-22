use ndarray::{Array1, ArrayView1, ArrayView2};
use crate::core::AnvilError;

/// Base trait for supervised models
pub trait Estimator<Y> {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<Y>) -> Result<(), AnvilError>;
}

/// Regression models (continuous output)
pub trait Regressor: Estimator<f64> {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, AnvilError>;
}

/// Classification models (discrete labels)
pub trait Classifier: Estimator<usize> {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, AnvilError>;
}

pub trait Transformer<I, O> {
    fn fit(&mut self, x: I) -> Result<(), AnvilError>;
    fn transform(&self, x: I) -> Result<O, AnvilError>;

    fn fit_transform(&mut self, x: I) -> Result<O, AnvilError>
    where
        I: Copy,
    {
        self.fit(x)?;
        self.transform(x)
    }
}
