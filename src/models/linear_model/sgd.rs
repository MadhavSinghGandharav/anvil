use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

use crate::{
    optim::{Optimizer, SGD},
    core::{Estimator, Regressor, AnvilError},
};

/// Linear regression using mini-batch SGD
pub struct SGDRegressor {
    params: Option<Array1<f64>>,
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

/// Builder
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: Box::new(SGD::new(0.01)),
        }
    }
}

impl Builder {
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    pub fn build(self) -> SGDRegressor {
        SGDRegressor {
            params: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
        }
    }
}

impl SGDRegressor {
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }

    /// # Errors
    /// - `NotFitted`
    pub fn weights(&self) -> Result<ArrayView1<'_, f64>, AnvilError> {
        let params = self.params.as_ref().ok_or(AnvilError::NotFitted)?;
        Ok(params.slice(s![1..]))
    }

    /// # Errors
    /// - `NotFitted`
    pub fn bias(&self) -> Result<f64, AnvilError> {
        Ok(self.params.as_ref().ok_or(AnvilError::NotFitted)?[0])
    }
}

impl Estimator<f64> for SGDRegressor {
    /// # Errors
    /// - `DimensionMismatch`
    /// - `InvalidParam`
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), AnvilError> {

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }


        let mut params = Array1::zeros(n_features + 1);
        let mut gradient = Array1::<f64>::zeros(n_features + 1);

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _ in 0..self.epochs {

            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {

                gradient.fill(0.0);

                let weights = params.slice(s![1..]);
                let bias = params[0];

                let mut grad_w = gradient.slice_mut(s![1..]);
                let mut grad_b = 0.0;

                for &idx in batch {

                    let row = x.row(idx);

                    let y_pred = row.dot(&weights) + bias;
                    let error = y[idx] - y_pred;
                    let grad = -error;

                    grad_b += grad;

                    for (g, &val) in grad_w.iter_mut().zip(row.iter()) {
                        *g += grad * val;
                    }
                }

                gradient[0] = grad_b;
                gradient *= 1.0 / batch.len() as f64;

                self.optimizer.step(
                    params.as_slice_mut().unwrap(),
                    gradient.as_slice().unwrap(),
                );
            }
        }

        self.params = Some(params);

        Ok(())
    }
}

impl Regressor for SGDRegressor {
    /// # Errors
    /// - `NotFitted`
    /// - `ShapeMismatch`
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<f64>, AnvilError> {

        let params = self.params.as_ref().ok_or(AnvilError::NotFitted)?;

        let weights = params.slice(s![1..]);
        let bias = params[0];

        if weights.len() != x.ncols() {
            return Err(AnvilError::ShapeMismatch {
                expected: weights.len(),
                got: x.ncols(),
                axis: "features",
            });
        }

        let mut preds = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {
            preds[i] = row.dot(&weights) + bias;
        }

        Ok(preds)
    }
}
