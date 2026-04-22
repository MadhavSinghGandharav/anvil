//! Stochastic Gradient Descent Regressor module.
//!
//! This module provides a linear regression implementation that optimizes 
//! parameters using mini-batch Stochastic Gradient Descent.

use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

use crate::{
    optim::{Optimizer, SGD},
    core::{Estimator, Regressor, AnvilError},
};

/// A Linear Regression model trained using mini-batch Stochastic Gradient Descent (SGD).
///
/// This regressor is suitable for large-scale regression tasks where traditional 
/// OLS (Ordinary Least Squares) might be computationally expensive.
///
/// # Examples
///
/// ```
/// use anvil::models::SGDRegressor;
/// use anvil::optim::SGD;
///
/// let model = SGDRegressor::builder()
///     .epochs(200)
///     .batch_size(16)
///     .optimizer(SGD::new(0.05))
///     .build();
/// ```
pub struct SGDRegressor {
    params: Option<Array1<f64>>,
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

/// A builder pattern implementation for configuring an [`SGDRegressor`].
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
    /// Sets the number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the size of mini-batches used in the optimization step.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    /// Sets the optimizer (e.g., SGD, Adam) for parameter updates.
    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Consumes the builder and returns a configured [`SGDRegressor`].
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
    /// Returns a new [`SGDRegressor`] with default settings.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a [`Builder`] to configure the regressor.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns the learned weights (coefficients) of the model.
    ///
    /// # Errors
    ///
    /// Returns [`AnvilError::NotFitted`] if the model has not been trained.
    pub fn weights(&self) -> Result<ArrayView1<'_, f64>, AnvilError> {
        let params = self.params.as_ref().ok_or(AnvilError::NotFitted)?;
        Ok(params.slice(s![1..]))
    }

    /// Returns the learned bias (intercept) of the model.
    ///
    /// # Errors
    ///
    /// Returns [`AnvilError::NotFitted`] if the model has not been trained.
    pub fn bias(&self) -> Result<f64, AnvilError> {
        Ok(self.params.as_ref().ok_or(AnvilError::NotFitted)?[0])
    }
}

impl Estimator<f64> for SGDRegressor {
    /// Fits the model to the training data $(X, y)$.
    ///
    /// # Errors
    ///
    /// * `AnvilError::DimensionMismatch`: If the number of samples in `x` and `y` do not match.
    /// * `AnvilError::InvalidParam`: If internal parameters cannot be converted to slices.
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
    /// Predicts target values for the given input samples.
    ///
    /// # Errors
    ///
    /// * `AnvilError::NotFitted`: If called before `fit`.
    /// * `AnvilError::ShapeMismatch`: If the feature count of `x` differs from the training data.
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
