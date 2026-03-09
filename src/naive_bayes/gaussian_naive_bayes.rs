use crate::core::DenseMatrix;
use crate::preprocessing::LabelEncoder;

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
/// - `var` → σ² for each `(class, feature)`
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
    /// `mean[c][j]` represents the mean of feature `j`
    /// for class `c`.
    mean: Option<DenseMatrix>,

    /// Variance matrix `(n_classes, n_features)`.
    ///
    /// Stores σ² for each class-feature pair.
    var: Option<DenseMatrix>,

    /// Precomputed Gaussian log constant:
    ///
    /// ```text
    /// -0.5 * ln(2πσ²)
    /// ```
    log_gauss_const: Option<DenseMatrix>,

    /// Precomputed inverse variance term:
    ///
    /// ```text
    /// 1 / (2σ²)
    /// ```
    inv_var: Option<DenseMatrix>,

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

impl GaussianNB {

    /// Returns a builder for constructing the classifier.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Constructs a new `GaussianNB` using default parameters.
    pub fn new() -> GaussianNB {
        Builder::default().build()
    }

    /// Fits the Gaussian Naive Bayes model.
    ///
    /// # Parameters
    ///
    /// - `features` → training matrix `(n_samples, n_features)`
    /// - `target` → class labels `(n_samples)`
    ///
    /// # Steps
    ///
    /// 1. Encode class labels using `LabelEncoder`.
    /// 2. Compute per-class:
    ///
    /// ```text
    /// mean
    /// variance
    /// ```
    ///
    /// 3. Apply variance smoothing.
    /// 4. Precompute Gaussian constants for faster prediction.
    /// 5. Compute class prior probabilities.
    ///
    /// # Panics
    ///
    /// - If number of samples in `features` and `target` mismatch.
    /// - If dataset contains zero samples.
    pub fn fit(&mut self, features: &DenseMatrix, target: &[usize]) {

        assert!(
            features.n_rows() == target.len(),
            "Number of samples mismatch"
        );

        assert!(
            features.n_rows() > 0,
            "Cannot fit with zero samples"
        );

        let mut encoder = LabelEncoder::new();
        let target = encoder.fit_transform(target);

        let n_classes = encoder.classes().len();
        let n_features = features.n_cols();

        self.classes = encoder.classes().to_vec();

        if let Some(priors) = &self.class_prob {
            assert!(
                priors.len() == n_classes,
                "Provided class_prob length must match number of classes"
            );
        }

        let mut sum = DenseMatrix::zeros(n_classes, n_features);
        let mut sum_sq = DenseMatrix::zeros(n_classes, n_features);
        let mut count = vec![0usize; n_classes];

        for i in 0..features.n_rows() {

            let c = target[i];
            count[c] += 1;

            let row = features.row(i);

            for j in 0..n_features {

                let x = row[j];

                *sum.get_mut(c, j) += x;
                *sum_sq.get_mut(c, j) += x * x;
            }
        }

        for c in 0..n_classes {

            let n = count[c] as f64;

            for j in 0..n_features {

                let mean = sum.get(c, j) / n;
                *sum.get_mut(c, j) = mean;

                let var = sum_sq.get(c, j) / n - mean * mean;

                *sum_sq.get_mut(c, j) = var;
            }
        }

        let eps = (self.var_smoothing * sum_sq.max()).max(self.var_smoothing);

        for v in sum_sq.as_slice_mut() {
            *v += eps;
        }

        let mut log_const = DenseMatrix::zeros(n_classes, n_features);
        let mut inv_var = DenseMatrix::zeros(n_classes, n_features);

        for c in 0..n_classes {

            for j in 0..n_features {

                let v = sum_sq.get(c, j);

                *log_const.get_mut(c, j) =
                    -0.5 * (2.0 * std::f64::consts::PI * v).ln();

                *inv_var.get_mut(c, j) =
                    1.0 / (2.0 * v);
            }
        }

        let priors = if let Some(ref p) = self.class_prob {
            p.clone()
        } else {

            let total = target.len() as f64;

            let mut priors = vec![0.0; n_classes];

            for c in target {
                priors[c] += 1.0;
            }

            for p in &mut priors {
                *p /= total;
            }

            priors
        };

        let log_priors: Vec<f64> =
            priors.iter().map(|p| p.ln()).collect();

        self.mean = Some(sum);
        self.var = Some(sum_sq);
        self.log_gauss_const = Some(log_const);
        self.inv_var = Some(inv_var);
        self.class_log_prior = Some(log_priors);
    }

    /// Predicts class labels for the given feature matrix.
    ///
    /// # Parameters
    ///
    /// - `features` → matrix `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// Vector containing predicted class labels.
    ///
    /// # Algorithm
    ///
    /// For each sample:
    ///
    /// ```text
    /// score_c =
    /// log(P(c))
    /// + Σ_j ( log_gauss_const[c,j]
    ///        - (x_j - μ_cj)² * inv_var[c,j] )
    /// ```
    ///
    /// The class with the **maximum score** is selected.
    ///
    /// # Panics
    ///
    /// - If model has not been fitted.
    /// - If feature dimension mismatch occurs.
    pub fn predict(&self, features: &DenseMatrix) -> Vec<usize> {

        let mean = self.mean.as_ref().expect("model not fitted");
        let log_const = self.log_gauss_const.as_ref().unwrap();
        let inv_var = self.inv_var.as_ref().unwrap();
        let log_prior = self.class_log_prior.as_ref().unwrap();

        assert!(
            features.n_cols() == mean.n_cols(),
            "Feature dimension mismatch"
        );

        let n_samples = features.n_rows();
        let n_classes = mean.n_rows();
        let n_features = mean.n_cols();

        let mut preds = Vec::with_capacity(n_samples);

        for i in 0..n_samples {

            let row = features.row(i);

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {

                let mut score = log_prior[c];

                for j in 0..n_features {

                    let x = row[j];
                    let m = mean.get(c, j);

                    score += log_const.get(c, j)
                        - (x - m) * (x - m) * inv_var.get(c, j);
                }

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            preds.push(self.classes[best_class]);
        }

        preds
    }
}

/// Builder for constructing `GaussianNB`.
///
/// Allows configuring optional parameters such as
/// class priors and variance smoothing.
pub struct Builder {
    class_prob: Option<Vec<f64>>,
    var_smoothing: f64,
}

impl Default for Builder {

    fn default() -> Self {
        Self {
            class_prob: None,
            var_smoothing: 1e-9,
        }
    }
}

impl Builder {

    /// Sets user-defined class prior probabilities.
    ///
    /// If provided, these values override
    /// automatically computed priors.
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    /// Sets the variance smoothing parameter.
    ///
    /// This value is used to stabilize variance estimates
    /// and prevent division by zero.
    pub fn var_smoothing(mut self, value: f64) -> Self {
        self.var_smoothing = value;
        self
    }

    /// Builds the `GaussianNB` classifier.
    pub fn build(self) -> GaussianNB {
        GaussianNB {
            mean: None,
            var: None,
            log_gauss_const: None,
            inv_var: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
            var_smoothing: self.var_smoothing,
        }
    }
}
