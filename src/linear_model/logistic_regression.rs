use crate::{
    core::{DenseMatrix, utils::dot},
    optim::{Optimizer, SGD},
    preprocessing::LabelEncoder,
};

use rand::seq::SliceRandom;

/// Logistic Regression classifier for binary classification.
///
/// The model learns a linear decision function:
///
///     f(x) = w^T x + b
///
/// and applies the sigmoid:
///
///     σ(z) = 1 / (1 + exp(-z))
///
/// The probability of the positive class is:
///
///     P(y=1 | x) = σ(w^T x + b)
///
/// # Loss
///
/// Logistic loss:
///
///     L = log(1 + exp(-y f(x)))
///
/// where `y ∈ {-1, 1}`.
pub struct LogisticRegression {

    /// Learned feature weights.
    weights: Option<Vec<f64>>,

    /// Bias term.
    bias: Option<[f64;1]>,

    /// Training epochs.
    epochs: usize,

    /// Mini batch size.
    batch_size: usize,

    /// Optimizer used for parameter updates.
    optimizer: Box<dyn Optimizer>,

    /// Original class labels.
    classes: [usize;2],
}

/// Builder for configuring [`LogisticRegression`].
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

    /// Create model with default parameters.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Return builder.
    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl LogisticRegression {

    /// Return learned weights.
    pub fn weights(&self) -> &[f64] {
        self.weights.as_ref().expect("Model not fitted")
    }

    /// Return learned bias.
    pub fn bias(&self) -> f64 {
        self.bias.as_ref().expect("Model not fitted")[0]
    }

    /// Return class labels.
    pub fn classes(&self) -> &[usize;2] {
        &self.classes
    }

    /// Predict probabilities of the positive class.
    ///
    /// Returns one probability per sample.
    ///
    /// # Panics
    ///
    /// Panics if model not fitted or dimension mismatch.
    pub fn predict_proba(&self, features: &DenseMatrix) -> Vec<f64> {

        let weights = self.weights.as_ref().expect("Model not fitted");
        let bias = self.bias.as_ref().expect("Model not fitted")[0];

        assert!(
            features.n_cols() == weights.len(),
            "Feature dimension mismatch"
        );

        let mut probs = Vec::with_capacity(features.n_rows());

        for i in 0..features.n_rows() {

            let row = features.row(i);
            let fx = dot(row, weights) + bias;

            probs.push(sigmoid(fx));
        }

        probs
    }

    /// Predict class labels.
    ///
    /// Returns a vector of predicted classes.
    pub fn predict(&self, features: &DenseMatrix) -> Vec<usize> {

        let probs = self.predict_proba(features);

        probs
            .into_iter()
            .map(|p| {
                if p >= 0.5 {
                    self.classes[1]
                } else {
                    self.classes[0]
                }
            })
            .collect()
    }

    /// Encode targets to `{-1,1}`.
    fn update_target(&mut self, target: &[usize]) -> Vec<f64> {

        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target);

        assert_eq!(
            encoder.classes().len(),
            2,
            "LogisticRegression supports only binary classification"
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

    /// Train the logistic regression model.
    ///
    /// Uses mini-batch gradient descent.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - samples != targets
    /// - batch_size == 0
    pub fn fit(&mut self, features: &DenseMatrix, target: &[usize]) {

        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        assert_eq!(
            n_samples,
            target.len(),
            "Number of samples and targets must match"
        );

        assert!(
            self.batch_size > 0,
            "batch_size must be greater than 0"
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

                    let yz = target[idx] * fx;

                    let sig = sigmoid(-yz);

                    for (g, &x) in gradient.iter_mut().zip(row) {
                        *g += -target[idx] * x * sig;
                    }

                    bias_gradient[0] += -target[idx] * sig;
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

    /// Set number of epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set mini-batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {

        assert!(
            batch_size > 0,
            "batch_size must be greater than 0"
        );

        self.batch_size = batch_size;
        self
    }

    /// Set custom optimizer.
    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Builder {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Build untrained model.
    pub fn build(self) -> LogisticRegression {

        LogisticRegression {
            weights: None,
            bias: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0;2],
        }
    }
}
