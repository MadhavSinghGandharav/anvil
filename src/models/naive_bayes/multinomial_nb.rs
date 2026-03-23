use crate::preprocessing::encoder::LabelEncoder;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

/// Multinomial Naive Bayes classifier.
///
/// Models each feature as a **word count** — suited for text classification
/// and other discrete count data.
///
/// The conditional log-likelihood is modeled as:
///
/// ```text
/// log P(x|c) = Σ_j x_j * log P(x_j | c)
/// ```
///
/// where `P(x_j | c)` is estimated with **Laplace smoothing**:
///
/// ```text
/// P(x_j | c) = (count(x_j, c) + 1) / (total_words(c) + n_features)
/// ```
///
/// # Stored Parameters
///
/// - `log_prob`        → `ln(P(x_j|c))` for each `(class, feature)`
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
pub struct MultinomialNB {

    /// Log probability of feature `j` given class `c`:
    ///
    /// ```text
    /// ln(P(x_j | c))
    /// ```
    log_prob: Option<Array2<f64>>,

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
}

/// Builder for configuring [`MultinomialNB`]
pub struct Builder {
    class_prob: Option<Vec<f64>>,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - class_prob = None (computed from class frequencies)
    fn default() -> Self {
        Self {
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

    /// Build model
    pub fn build(self) -> MultinomialNB {
        MultinomialNB {
            log_prob: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
        }
    }
}

impl MultinomialNB {

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

    /// Fits the Multinomial Naive Bayes model
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

        assert!(features.iter().all(|&x| x > 0.0),
            "Negative values found"
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

        // Loop 1: accumulate per-class feature counts and total word counts
        let mut sum          = Array2::<f64>::zeros((n_classes, n_features));
        let mut total_words  = vec![0.0f64; n_classes];
        let mut sample_count = vec![0usize; n_classes];

        for (i, row) in features.outer_iter().enumerate() {

            let c = target[i];
            sample_count[c] += 1;

            Zip::from(sum.row_mut(c))
                .and(&row)
                .for_each(|s, &x| *s += x);

            // row.sum() is SIMD-accelerated — cheaper than accumulating inside Zip
            total_words[c] += row.sum();
        }

        // Loop 2: apply Laplace smoothing and compute log probabilities
        let mut log_prob = Array2::<f64>::zeros((n_classes, n_features));

        for c in 0..n_classes {

            // Laplace denominator: total_words(c) + n_features
            let n = total_words[c] + n_features as f64;

            Zip::from(log_prob.row_mut(c))
                .and(sum.row(c))
                .for_each(|lp, &s| {
                    *lp = ((s + 1.0) / n).ln();
                });
        }

        // compute class log-prior probabilities
        let total = target.len() as f64;

        let log_priors: Vec<f64> = if let Some(ref p) = self.class_prob {
            p.iter().map(|p| p.ln()).collect()
        } else {
            sample_count.iter().map(|&c| (c as f64 / total).ln()).collect()
        };

        self.log_prob        = Some(log_prob);
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

        let log_prob  = self.log_prob.as_ref().expect("Model not fitted");
        let log_prior = self.class_log_prior.as_ref().expect("Model not fitted");

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

            // log-space score: log P(c) + x · log_prob[c]
            for c in 0..n_classes {

                let mut score = log_prior[c];

                Zip::from(&row)
                    .and(log_prob.row(c))
                    .for_each(|&x, &lp| score += x * lp);

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
