//! Perceptron module for binary classification.
//!
//! This module provides an implementation of the classic Perceptron algorithm,
//! a fundamental linear classifier that updates its weights based on misclassified samples.

use crate::{
    optim::{Optimizer, SGD},
    preprocessing::encoder::LabelEncoder,
    core::{Estimator, Classifier, AnvilError},
};
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

/// A Perceptron classifier for binary classification.
///
/// The Perceptron learns a linear decision boundary by iteratively adjusting 
/// weights whenever a training instance is misclassified ($y \cdot f(x) \le 0$).
///
/// # Examples
///
/// ```
/// use anvil::models::Perceptron;
///
/// let model = Perceptron::builder()
///     .epochs(100)
///     .batch_size(1) // Traditional online learning
///     .build();
/// ```
pub struct Perceptron {
    params: Option<Array1<f64>>,
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
    classes: [usize; 2],
}

/// A builder pattern implementation for configuring a [`Perceptron`] model.
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
    /// Sets the number of training iterations (epochs) over the entire dataset.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the batch size for weight updates.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    /// Sets the optimizer to be used during the fitting process.
    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Consumes the builder and returns a configured [`Perceptron`] instance.
    pub fn build(self) -> Perceptron {
        Perceptron {
            params: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0; 2],
        }
    }
}

impl Perceptron {
    /// Returns a new instance of [`Perceptron`] with default settings.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a [`Builder`] to configure the model.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Encodes target labels into numeric values $\{-1.0, 1.0\}$.
    ///
    /// # Errors
    ///
    /// Returns `AnvilError::InvalidParam` if more or fewer than 2 classes are detected.
    fn update_target(&mut self, target: &[usize]) -> Result<Vec<f64>, AnvilError> {
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target);

        if encoder.classes().len() != 2 {
            return Err(AnvilError::InvalidParam {
                param: "y",
                reason: "Perceptron supports only binary classification".into(),
            });
        }

        self.classes = [encoder.classes()[0], encoder.classes()[1]];

        Ok(
            encoded
                .into_iter()
                .map(|i| if i == 0 { -1.0 } else { 1.0 })
                .collect()
        )
    }
}

impl Estimator<usize> for Perceptron {
    /// Trains the Perceptron model on the provided dataset.
    ///
    /// # Errors
    ///
    /// * `AnvilError::DimensionMismatch`: If `x` and `y` have different sample counts.
    /// * `AnvilError::InvalidParam`: If `y` is not contiguous or `batch_size` is 0.
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

        let y_slice = y.as_slice().ok_or(AnvilError::InvalidParam {
            param: "y",
            reason: "not contiguous".into(),
        })?;

        let target = self.update_target(y_slice)?;

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
                    let fx = row.dot(&weights) + bias;

                    // Perceptron update rule: update if misclassified
                    if y * fx <= 0.0 {
                        grad_b -= y;

                        for (g, &val) in grad_w.iter_mut().zip(row.iter()) {
                            *g -= y * val;
                        }
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

impl Classifier for Perceptron {
    /// Predicts class labels for a set of samples.
    ///
    /// # Errors
    ///
    /// * `AnvilError::NotFitted`: If called before `fit`.
    /// * `AnvilError::ShapeMismatch`: If the number of features in `x` differs from the training data.
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
            let fx = row.dot(&weights) + bias;

            preds[i] = if fx >= 0.0 {
                self.classes[1]
            } else {
                self.classes[0]
            };
        }

        Ok(preds)
    }
}
