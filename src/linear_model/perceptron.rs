use crate::{
    core::{DenseMatrix, utils::dot},
    optim::{Optimizer, SGD},
    preprocessing::LabelEncoder,
};

use rand::seq::SliceRandom;

/// Linear binary classifier trained using the **Perceptron algorithm**.
///
/// The perceptron learns a **linear decision boundary** of the form:
///
/// ```text
/// f(x) = wᵀx + b
/// ```
///
/// where:
///
/// - `w` = weight vector  
/// - `b` = bias (intercept)
///
/// A prediction is made using the sign of the decision function:
///
/// ```text
/// ŷ = sign(wᵀx + b)
/// ```
///
/// # Training Rule
///
/// For each sample `(x, y)` where `y ∈ {-1, 1}`:
///
/// ```text
/// if y * (wᵀx + b) ≤ 0:
///     w ← w - η (-y x)
///     b ← b - η (-y)
/// ```
///
/// where `η` is the learning rate.
///
/// # Notes
///
/// - This implementation supports **binary classification only**.
/// - Target labels are automatically encoded to `{-1, 1}`.
/// - Convergence is guaranteed only if the dataset is **linearly separable**.
///
/// # Mini-Batch Training
///
/// When `batch_size = 1`, the algorithm behaves as the **classic online perceptron**.
///
/// When `batch_size > 1`, gradients are averaged across the batch and updated using
/// the configured optimizer.
///
/// # Example
///
/// ```ignore
/// let mut model = Perceptron::builder()
///     .epochs(200)
///     .batch_size(1)
///     .build();
///
/// model.fit(&x, &y);
///
/// let preds = model.predict(&x);
/// ```
pub struct Perceptron {

    /// Learned feature weights.
    ///
    /// This vector has length equal to the number of features.
    ///
    /// Initialized during [`fit`].
    weights: Option<Vec<f64>>,

    /// Learned bias (intercept).
    ///
    /// Stored as a length-1 array so the optimizer can update
    /// it using the same interface as weights.
    bias: Option<[f64;1]>,

    /// Number of training epochs.
    epochs: usize,

    /// Mini-batch size used during training.
    batch_size: usize,

    /// Optimizer used for updating parameters.
    optimizer: Box<dyn Optimizer>,

    /// Original class labels.
    ///
    /// The perceptron internally uses `{-1,1}` labels but predictions
    /// are mapped back to these original class values.
    classes: [usize;2],
}

/// Builder for configuring a [`Perceptron`] classifier.
///
/// Allows customization of:
///
/// - training epochs
/// - batch size
/// - optimizer
///
/// # Defaults
///
/// - `epochs = 100`
/// - `batch_size = 1`
/// - `optimizer = SGD(learning_rate = 1.0)`
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {

    /// Creates a builder with default hyperparameters.
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: Box::new(SGD::new(1.0)),
        }
    }
}

impl Perceptron {

    /// Creates a perceptron classifier with default parameters.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// Perceptron::builder().build()
    /// ```
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a builder used to configure the model.
    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl Perceptron {

    /// Returns the learned weight vector.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been trained.
    pub fn weights(&self) -> &[f64] {
        self.weights.as_ref().expect("Model not fitted")
    }

    /// Returns the learned bias value.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been trained.
    pub fn bias(&self) -> f64 {
        self.bias.as_ref().expect("Model not fitted")[0]
    }

    /// Returns the original class labels.
    pub fn classes(&self) -> &[usize;2] {
        &self.classes
    }

    /// Predict class labels for a feature matrix.
    ///
    /// Each row of `features` is treated as a separate sample.
    ///
    /// # Returns
    ///
    /// A vector containing predicted class labels.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - the model has not been fitted
    /// - the number of features does not match the trained model
    pub fn predict(&self, features: &DenseMatrix) -> Vec<usize> {

        let weights = self.weights.as_ref().expect("Model not fitted");
        let bias = self.bias.as_ref().expect("Model not fitted")[0];

        assert!(
            features.n_cols() == weights.len(),
            "Feature dimension mismatch"
        );

        let mut preds = Vec::with_capacity(features.n_rows());

        for i in 0..features.n_rows() {

            let row = features.row(i);
            let fx = dot(row, weights) + bias;

            if fx >= 0.0 {
                preds.push(self.classes[1]);
            } else {
                preds.push(self.classes[0]);
            }
        }

        preds
    }

    /// Encodes target labels to `{-1, 1}`.
    ///
    /// The original class labels are stored in [`classes`].
    ///
    /// # Panics
    ///
    /// Panics if the dataset contains more than two classes.
    fn update_target(&mut self, target: &[usize]) -> Vec<f64> {

        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target);

        assert_eq!(
            encoder.classes().len(),
            2,
            "Perceptron supports only binary classification"
        );

        self.classes = [
            encoder.classes()[0],
            encoder.classes()[1],
        ];

        encoded
            .into_iter()
            .map(|i| if i == 0 { -1.0 } else { 1.0 })
            .collect()
    }

    /// Train the perceptron classifier.
    ///
    /// Uses **mini-batch stochastic subgradient descent**.
    ///
    /// # Parameters
    ///
    /// - `features` — input feature matrix
    /// - `target` — class labels
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - number of samples does not match target length
    /// - `batch_size == 0`
    pub fn fit(&mut self, features: &DenseMatrix, target: &[usize]) {

        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        assert_eq!(
            n_samples,
            target.len(),
            "Number of samples and targets must match"
        );

        let mut weights = vec![0.0; n_features];
        let mut bias = [0.0];

        let target = self.update_target(target);

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let mut gradient = vec![0.0; n_features];
        let mut bias_gradient = [0.0];

        for _ in 0..self.epochs {

            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {

                gradient.fill(0.0);
                bias_gradient[0] = 0.0;

                for &idx in batch {

                    let row = features.row(idx);
                    let fx = dot(row, &weights) + bias[0];

                    if target[idx] * fx <= 0.0 {

                        for (g, &x) in gradient.iter_mut().zip(row) {
                            *g += -target[idx] * x;
                        }

                        bias_gradient[0] += -target[idx];
                    }
                }

                let inv_bs = 1.0 / batch.len() as f64;

                for g in &mut gradient {
                    *g *= inv_bs;
                }

                bias_gradient[0] *= inv_bs;

                self.optimizer.step(&mut weights, &gradient);
                self.optimizer.step(&mut bias, &bias_gradient);
            }
        }

        self.weights = Some(weights);
        self.bias = Some(bias);
    }
}

impl Builder {

    /// Set the number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set mini-batch size.
    ///
    /// `batch_size = 1` corresponds to the classical online perceptron.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size == 0`.
    pub fn batch_size(mut self, batch_size: usize) -> Self {

        assert!(
            batch_size > 0,
            "batch_size must be greater than 0"
        );

        self.batch_size = batch_size;
        self
    }

    /// Use a custom optimizer.
    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Builder {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Build an untrained perceptron model.
    ///
    /// Parameters are initialized during [`fit`].
    pub fn build(self) -> Perceptron {

        Perceptron {
            weights: None,
            bias: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0;2],
        }
    }
}
