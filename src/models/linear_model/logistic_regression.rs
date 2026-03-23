use crate::{
    optim::{Optimizer, SGD},
    preprocessing::encoder::LabelEncoder,
};
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

/// Linear binary classifier trained using **Logistic Regression**.
///
/// The model learns a linear decision boundary:
///
/// f(x) = wᵀx + b
///
/// and optimizes the **logistic loss**:
///
/// L = log(1 + exp(-y(wᵀx + b)))
///
/// where `y ∈ {-1, 1}`.
///
/// # Notes
///
/// - Supports **binary classification only**
/// - Target labels are internally converted to `{-1,1}`
/// - Uses **mini-batch stochastic gradient descent**
pub struct LogisticRegression {
    /// Learned parameters `[bias, weights...]`
    params: Option<Array1<f64>>,

    /// Number of training epochs
    epochs: usize,

    /// Mini-batch size
    batch_size: usize,

    /// Optimizer used for parameter updates
    optimizer: Box<dyn Optimizer>,

    /// Original class labels
    classes: [usize; 2],
}

/// Builder for configuring [`LogisticRegression`]
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - epochs = 100
    /// - batch_size = 1
    /// - optimizer = SGD(1.0)
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: Box::new(SGD::new(1.0)),
        }
    }
}

impl Builder {

    /// Set number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set mini-batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    /// Set optimizer
    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Build model
    pub fn build(self) -> LogisticRegression {
        LogisticRegression {
            params: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0; 2],
        }
    }
}

/// Numerically stable sigmoid: avoids overflow for large positive or negative `z`
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

    /// Create model with default configuration
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns builder
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns learned weights
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted
    pub fn weights(&self) -> ArrayView1<'_, f64> {
        let params = self.params.as_ref().expect("Model not fitted");
        params.slice(s![1..])
    }

    /// Returns learned bias
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted
    pub fn bias(&self) -> f64 {
        self.params.as_ref().expect("Model not fitted")[0]
    }

    /// Returns the two original class labels in ascending order
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted
    pub fn classes(&self) -> &[usize; 2] {
        &self.classes
    }

    /// Encodes `target` labels as `{-1, 1}` and stores the original classes
    fn update_target(&mut self, target: &[usize]) -> Vec<f64> {

        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target);

        assert_eq!(
            encoder.classes().len(),
            2,
            "LogisticRegression supports only binary classification"
        );

        self.classes = [encoder.classes()[0], encoder.classes()[1]];

        encoded
            .into_iter()
            .map(|i| if i == 0 { -1.0 } else { 1.0 })
            .collect()
    }

    /// Fits the model using mini-batch stochastic gradient descent
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - features and target size mismatch
    /// - target does not contain exactly 2 distinct classes
    /// - batch_size == 0
    pub fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<usize>) {

        let n_samples = features.nrows();
        let n_features = features.ncols();

        assert_eq!(
            n_samples,
            target.len(),
            "Number of samples and target values must match"
        );

        assert!(
            self.batch_size > 0,
            "batch_size must be greater than 0"
        );

        // initialize parameters [bias, weights...]
        let mut params = Array1::<f64>::zeros(n_features + 1);

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let target = self.update_target(target.as_slice().unwrap());

        let mut gradient = Array1::<f64>::zeros(n_features + 1);

        for _ in 0..self.epochs {

            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {

                gradient.fill(0.0);

                // slice once per batch
                let weights = params.slice(s![1..]);
                let bias = params[0];

                let mut grad_w = gradient.slice_mut(s![1..]);
                let mut grad_b = 0.0;

                for &idx in batch {

                    let row = features.row(idx);

                    // logistic loss gradient: -y * sigmoid(-y * z)
                    let y = target[idx];
                    let z = row.dot(&weights) + bias;
                    let coeff = -y * sigmoid(-y * z);

                    // bias gradient
                    grad_b += coeff;

                    // weight gradients
                    for (g, &x) in grad_w.iter_mut().zip(row.iter()) {
                        *g += coeff * x;
                    }
                }

                gradient[0] = grad_b;

                // average gradient
                let inv_bs = 1.0 / batch.len() as f64;
                gradient *= inv_bs;

                self.optimizer
                    .step(params.as_slice_mut().unwrap(), gradient.as_slice().unwrap());
            }
        }

        self.params = Some(params);
    }

    /// Predict class probabilities (probability of the positive class)
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - model not fitted
    /// - feature dimension mismatch
    pub fn predict_proba(&self, features: ArrayView2<f64>) -> Array1<f64> {

        let params = self.params.as_ref().expect("Model not fitted");

        let weights = params.slice(s![1..]);
        let bias = params[0];

        assert!(
            weights.len() == features.ncols(),
            "Feature dimension mismatch"
        );

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {
            // sigmoid maps the decision function to [0, 1]
            preds[i] = sigmoid(row.dot(&weights) + bias);
        }

        preds
    }

    /// Predict target labels
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - model not fitted
    /// - feature dimension mismatch
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<usize> {

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
}
