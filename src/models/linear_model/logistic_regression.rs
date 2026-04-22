//! Logistic Regression module for binary classification.
//!
//! This module provides a flexible implementation of Logistic Regression using
//! gradient-based optimization.

use crate::{
    optim::{Optimizer, SGD},
    preprocessing::encoder::LabelEncoder,
    core::{Estimator, Classifier, AnvilError,Transformer},
};
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

/// A Logistic Regression classifier for binary classification tasks.
///
/// It uses a sigmoid activation function and can be configured with various
/// optimizers (e.g., SGD) and batch sizes for training.
pub struct LogisticRegression {
    params: Option<Array1<f64>>,
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
    classes: [usize; 2],
}

/// Builder for configuring [`LogisticRegression`]
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
            optimizer: Box::new(SGD::new(1.0)),
        }
    }
}

impl Builder {
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// # Errors
    /// Returns error if batch_size == 0 (validated at fit time)
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    pub fn build(self) -> Result<LogisticRegression, AnvilError> {

        if self.batch_size == 0 {
            return Err(AnvilError::InvalidParam {
                param: "batch_size",
                reason: "must be > 0".into()
            });
        }

        if self.epochs == 0 {
            return Err(AnvilError::InvalidParam {
                param: "epochs",
                reason: "must be > 0".into()
            });
        }

        Ok(LogisticRegression {
            params: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0; 2],
        })
    }
}

/// Numerically stable sigmivate, no user-facing invariants to check there) — all validation is front-loaded in fit.oid
#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let exp_neg = (-z).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = z.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

impl LogisticRegression {
    pub fn new() -> Result<Self,AnvilError> {
        Self::builder().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Encodes labels into {-1, 1}
    ///
    /// # Errors
    /// - `InvalidParam` if not binary classification
    fn update_target(&mut self, target: ArrayView1<usize>) -> Result<Vec<f64>, AnvilError> {
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target)?;
        
        let classes = encoder.classes()?;
        if classes.len() != 2 {
            return Err(AnvilError::InvalidParam {
                param: "y",
                reason: "LogisticRegression supports only binary classification".into(),
            });
        }

        self.classes = [classes[0],classes[1]];

        Ok(
            encoded
                .into_iter()
                .map(|i| if i == 0 { -1.0 } else { 1.0 })
                .collect()
        )
    }
}

impl Estimator<usize> for LogisticRegression {
    /// # Errors
    /// - `DimensionMismatch` if X and y size mismatch
    /// - `InvalidParam` if batch_size == 0
    /// - `InvalidParam` if y not contiguous
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<usize>,
    ) -> Result<(), AnvilError> {

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }

        if self.batch_size == 0 {
            return Err(AnvilError::InvalidParam {
                param: "batch_size",
                reason: "must be > 0".into(),
            });
        }
 

        let target = self.update_target(y)?;

        let mut params = Array1::<f64>::zeros(n_features + 1);
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

                    let y = target[idx];
                    let z = row.dot(&weights) + bias;
                    let coeff = -y * sigmoid(-y * z);

                    grad_b += coeff;

                    for (g, &val) in grad_w.iter_mut().zip(row.iter()) {
                        *g += coeff * val;
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

impl Classifier for LogisticRegression {
    /// # Errors
    /// - `NotFitted` if model not trained
    /// - `ShapeMismatch` if feature mismatch
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<usize>, AnvilError> {

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
            let prob = sigmoid(row.dot(&weights) + bias);

            preds[i] = if prob >= 0.5 {
                self.classes[1]
            } else {
                self.classes[0]
            };
        }

        Ok(preds)
    }
}
