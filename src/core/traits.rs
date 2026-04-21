use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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

/// Transformers (scalers, PCA, etc.)
pub trait Transformer {
    fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), AnvilError>;

    fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, AnvilError>;

    fn fit_transform(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>, AnvilError> {
        self.fit(x)?;
        self.transform(x)
    }
}
