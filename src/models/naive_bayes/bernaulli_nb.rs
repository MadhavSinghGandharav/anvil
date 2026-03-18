use crate::preprocessing::LabelEncoder;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

/// Bernoulli Naive Bayes classifier.
///
/// Assumes each feature is **binary** — either present (`1`) or absent (`0`).
/// A configurable `threshold` binarizes continuous inputs during `fit`.
///
/// The conditional log-likelihood is modeled as:
///
/// ```text
/// log P(x|c) = Σ_j [ x_j * log P(x_j=1|c)
///                   + (1 - x_j) * log P(x_j=0|c) ]
/// ```
///
/// where `P(x_j=1|c)` is estimated with **Laplace smoothing** to
/// prevent zero-probability issues:
///
/// ```text
/// P(x_j=1|c) = (count(x_j=1, c) + 1) / (count(c) + 2)
/// ```
///
/// # Stored Parameters
///
/// - `log_prob`        → `ln(P(x_j=1|c))` for each `(class, feature)`
/// - `log_prob_neg`    → `ln(P(x_j=0|c))` for each `(class, feature)`
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
pub struct BernoulliNB {

    /// Log probability of feature being present per class:
    ///
    /// ```text
    /// ln(P(x_j=1|c))
    /// ```
    log_prob: Option<Array2<f64>>,

    /// Log probability of feature being absent per class:
    ///
    /// ```text
    /// ln(P(x_j=0|c))
    /// ```
    log_prob_neg: Option<Array2<f64>>,

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

    /// Binarization threshold.
    ///
    /// Features above this value are treated as `1`, others as `0`.
    threshold: f64,
}

/// Builder for configuring [`BernoulliNB`]
pub struct Builder {
    threshold: f64,
    class_prob: Option<Vec<f64>>,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - threshold = 0.0
    /// - class_prob = None (computed from class frequencies)
    fn default() -> Self {
        Self {
            threshold: 0.0,
            class_prob: None,
        }
    }
}

impl Builder {

    /// Set user-defined class prior probabilities
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    /// Set binarization threshold
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Build model
    pub fn build(self) -> BernoulliNB {
        BernoulliNB {
            log_prob: None,
            log_prob_neg: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
            threshold: self.threshold,
        }
    }
}

impl BernoulliNB {

    /// Create model with default configuration
    pub fn new() -> Self {
        Builder::default().build()
    }

    /// Returns builder
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns the original class labels
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Fits the Bernoulli Naive Bayes model
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

        let n_classes  = encoder.classes().len();
        let n_features = features.ncols();

        self.classes = encoder.classes().to_vec();

        if let Some(priors) = &self.class_prob {
            assert!(
                priors.len() == n_classes,
                "Provided class_prob length must match number of classes"
            );
        }

        // Loop 1: accumulate per-class binary feature counts
        let mut count       = Array2::<f64>::zeros((n_classes, n_features));
        let mut class_count = vec![0usize; n_classes];

        for (i, row) in features.outer_iter().enumerate() {

            let c = target[i];
            class_count[c] += 1;

            Zip::from(count.row_mut(c))
                .and(&row)
                .for_each(|cnt, &x| {
                    if x > self.threshold {
                        *cnt += 1.0;
                    }
                });
        }

        // Loop 2: apply Laplace smoothing and compute log probabilities
        let mut log_prob     = Array2::<f64>::zeros((n_classes, n_features));
        let mut log_prob_neg = Array2::<f64>::zeros((n_classes, n_features));

        for c in 0..n_classes {

            // Laplace denominator: count(c) + 2
            let n = class_count[c] as f64 + 2.0;

            Zip::from(log_prob.row_mut(c))
                .and(log_prob_neg.row_mut(c))
                .and(count.row(c))
                .for_each(|lp, lp_neg, &cnt| {
                    let prob = (cnt + 1.0) / n;
                    *lp     = prob.ln();
                    *lp_neg = (1.0 - prob).ln();
                });
        }

        // compute class log-prior probabilities
        let total = target.len() as f64;

        let log_priors: Vec<f64> = if let Some(ref p) = self.class_prob {
            p.iter().map(|p| p.ln()).collect()
        } else {
            class_count.iter().map(|&c| (c as f64 / total).ln()).collect()
        };

        self.log_prob        = Some(log_prob);
        self.log_prob_neg    = Some(log_prob_neg);
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

        let log_prob     = self.log_prob.as_ref().expect("Model not fitted");
        let log_prob_neg = self.log_prob_neg.as_ref().expect("Model not fitted");
        let log_prior    = self.class_log_prior.as_ref().expect("Model not fitted");

        assert_eq!(
            features.ncols(),
            log_prob.ncols(),
            "Feature dimension mismatch"
        );

        let n_classes = log_prob.nrows();

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            // log-space score: log P(c) + Σ_j log P(x_j | c)
            for c in 0..n_classes {

                let mut score = log_prior[c];

                Zip::from(&row)
                    .and(log_prob.row(c))
                    .and(log_prob_neg.row(c))
                    .for_each(|&x, &lp, &lp_neg| {
                        // binarize and select appropriate log probability
                        if x > self.threshold {
                            score += lp;
                        } else {
                            score += lp_neg;
                        }
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
