use crate::preprocessing::LabelEncoder;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use std::f64::consts::PI;

/// Gaussian Naive Bayes classifier.
///
/// This implementation assumes that **each feature follows a Gaussian
/// (normal) distribution within each class**.
///
/// The conditional likelihood is modeled as:
///
/// ```text
/// P(x_j | c) = 1 / sqrt(2πσ²) * exp(-(x_j - μ)² / (2σ²))
/// ```
///
/// During prediction we operate in **log-space** to avoid floating
/// underflow:
///
/// ```text
/// log P(x|c) = log P(c) + Σ_j log P(x_j | c)
/// ```
///
/// This implementation also **precomputes Gaussian constants**
/// during `fit()` to speed up prediction.
///
/// # Stored Parameters
///
/// - `mean` → μ for each `(class, feature)`
/// - `log_gauss_const` → `-0.5 * ln(2πσ²)`
/// - `inv_var` → `1 / (2σ²)`
/// - `class_log_prior` → `ln(P(c))`
///
/// # Complexity
///
/// Training:
///
/// ```text
/// O(samples × features)
/// ```
///
/// Prediction:
///
/// ```text
/// O(samples × classes × features)
/// ```
///
/// # Numerical Stability
///
/// Variance smoothing is applied to prevent zero-variance issues:
///
/// ```text
/// var += max(var_smoothing * max(var), var_smoothing)
/// ```
pub struct GaussianNB {

    /// Mean matrix of shape `(n_classes, n_features)`.
    ///
    /// `mean[[c, j]]` represents the mean of feature `j`
    /// for class `c`.
    mean: Option<Array2<f64>>,

    /// Precomputed Gaussian log constant:
    ///
    /// ```text
    /// -0.5 * ln(2πσ²)
    /// ```
    log_gauss_const: Option<Array2<f64>>,

    /// Precomputed inverse variance term:
    ///
    /// ```text
    /// 1 / (2σ²)
    /// ```
    inv_var: Option<Array2<f64>>,

    /// Log prior probability of each class:
    ///
    /// ```text
    /// ln(P(c))
    /// ```
    class_log_prior: Option<Vec<f64>>,

    /// Optional user-provided class probabilities.
    ///
    /// If `None`, priors are computed from class frequencies.
    class_prob: Option<Vec<f64>>,

    /// Original class labels.
    ///
    /// Needed because `LabelEncoder` converts labels
    /// into contiguous indices `[0..n_classes)`.
    classes: Vec<usize>,

    /// Variance smoothing parameter used to stabilize
    /// Gaussian likelihood computation.
    var_smoothing: f64,
}

/// Builder for configuring [`GaussianNB`]
pub struct Builder {
    class_prob: Option<Vec<f64>>,
    var_smoothing: f64,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - class_prob = None (computed from class frequencies)
    /// - var_smoothing = 1e-9
    fn default() -> Self {
        Self {
            class_prob: None,
            var_smoothing: 1e-9,
        }
    }
}

impl Builder {

    /// Set user-defined class prior probabilities
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    /// Set variance smoothing parameter
    pub fn var_smoothing(mut self, value: f64) -> Self {
        self.var_smoothing = value;
        self
    }

    /// Build model
    pub fn build(self) -> GaussianNB {
        GaussianNB {
            mean: None,
            log_gauss_const: None,
            inv_var: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
            var_smoothing: self.var_smoothing,
        }
    }
}

impl GaussianNB {

    /// Create model with default configuration
    pub fn new() -> Self {
        Builder::default().build()
    }

    /// Returns builder
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns classes
    pub fn classes(&self) -> &Vec<usize>{
        &self.classes
    }

    /// Fits the Gaussian Naive Bayes model
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - features and target size mismatch
    /// - dataset contains zero samples
    /// - `class_prob` length does not match number of classes
    pub fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<usize>) {

        assert!(
            features.nrows() == target.len(),
            "Number of samples mismatch"
        );

        assert!(
            features.nrows() > 0,
            "Cannot fit with zero samples"
        );

        // encode class labels into contiguous indices
        let mut encoder = LabelEncoder::new();
        let target = encoder.fit_transform(target.as_slice().unwrap());

        let n_classes = encoder.classes().len();
        let n_features = features.ncols();

        self.classes = encoder.classes().to_vec();

        if let Some(priors) = &self.class_prob {
            assert!(
                priors.len() == n_classes,
                "Provided class_prob length must match number of classes"
            );
        }

        // Loop 1: accumulate per-class feature sums and squared sums
        let mut sum    = Array2::<f64>::zeros((n_classes, n_features));
        let mut sum_sq = Array2::<f64>::zeros((n_classes, n_features));
        let mut count  = vec![0usize; n_classes];

        for (i, &c) in target.iter().enumerate() {

            count[c] += 1;

            let row = features.row(i);

            Zip::from(sum.row_mut(c))
                .and(sum_sq.row_mut(c))
                .and(&row)
                .for_each(|s, sq, &x| {
                    *s  += x;
                    *sq += x * x;
                });
        }

        // Loop 2: compute mean, raw variance, track max_var
        let mut max_var = 0.0_f64;

        for c in 0..n_classes {

            let n = count[c] as f64;

            Zip::from(sum.row_mut(c))
                .and(sum_sq.row_mut(c))
                .for_each(|s, sq| {
                    let mean = *s / n;
                    let var  = *sq / n - mean * mean;
                    *s  = mean;
                    *sq = var;
                    if var > max_var { max_var = var; }
                });
        }

        // Loop 3: smooth variance, precompute log_gauss_const and inv_var
        let eps = (self.var_smoothing * max_var).max(self.var_smoothing);

        let mut log_const = Array2::<f64>::zeros((n_classes, n_features));
        let mut inv_var   = Array2::<f64>::zeros((n_classes, n_features));

        Zip::from(&sum_sq)
            .and(&mut log_const)
            .and(&mut inv_var)
            .for_each(|&var, lc, iv| {
                let v = var + eps;
                *lc = -0.5 * (2.0 * PI * v).ln();
                *iv =  1.0 / (2.0 * v);
            });

        // compute class log-prior probabilities
        let priors = if let Some(ref p) = self.class_prob {
            p.clone()
        } else {

            let total = target.len() as f64;

            let mut priors = vec![0.0f64; n_classes];

            for &c in &target {
                priors[c] += 1.0;
            }

            for p in &mut priors {
                *p /= total;
            }

            priors
        };

        let log_priors: Vec<f64> = priors.iter().map(|p| p.ln()).collect();

        self.mean            = Some(sum);
        self.log_gauss_const = Some(log_const);
        self.inv_var         = Some(inv_var);
        self.class_log_prior = Some(log_priors);
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

        let mean      = self.mean.as_ref().expect("Model not fitted");
        let log_const = self.log_gauss_const.as_ref().expect("Model not fitted");
        let inv_var   = self.inv_var.as_ref().expect("Model not fitted");
        let log_prior = self.class_log_prior.as_ref().expect("Model not fitted");

        assert!(
            features.ncols() == mean.ncols(),
            "Feature dimension mismatch"
        );

        let n_classes = mean.nrows();

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            // log-space score: log P(c) + Σ_j log P(x_j | c)
            for c in 0..n_classes {

                let mut score = log_prior[c];

                Zip::from(&row)
                    .and(mean.row(c))
                    .and(log_const.row(c))
                    .and(inv_var.row(c))
                    .for_each(|&x, &m, &lc, &iv| {
                        score += lc - (x - m) * (x - m) * iv;
                    });

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            preds[i] = self.classes[best_class];
        }

        preds
    }
}
