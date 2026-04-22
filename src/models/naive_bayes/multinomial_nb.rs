//! Multinomial Naive Bayes module.
//!
//! This module implements the Multinomial Naive Bayes algorithm, which is widely used 
//! for document classification (e.g., spam filtering) where features represent 
//! word counts or frequencies.

use crate::{
    preprocessing::encoder::LabelEncoder,
    core::{Estimator, Classifier, AnvilError, Transformer},
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

/// Multinomial Naive Bayes (MultinomialNB) classifier.
///
/// This model is intended for use with discrete data (e.g., word counts for text). 
/// It requires non-negative feature values and uses additive (Laplace) smoothing 
/// to handle features not present in the training set.
///
/// # Examples
///
/// ```
/// use anvil::models::MultinomialNB;
///
/// let model = MultinomialNB::builder()
///     .build()
///     .unwrap();
/// ```
pub struct MultinomialNB {
    log_prob: Option<Array2<f64>>,
    class_log_prior: Option<Vec<f64>>,
    class_prob: Option<Vec<f64>>,
    classes: Vec<usize>,
}

/// A builder for configuring and creating a [`MultinomialNB`] instance.
pub struct Builder {
    class_prob: Option<Vec<f64>>,
}

impl Default for Builder {
    fn default() -> Self {
        Self { class_prob: None }
    }
}

impl Builder {
    /// Sets user-defined prior probabilities for the classes.
    ///
    /// If provided, these values must be non-negative and sum to a positive value.
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    /// Consumes the builder and returns a [`MultinomialNB`] instance.
    ///
    /// # Errors
    ///
    /// Returns [`AnvilError::InvalidParam`] if `class_prob` is empty, contains 
    /// negative values, or sums to zero.
    pub fn build(self) -> Result<MultinomialNB, AnvilError> {

        if let Some(ref probs) = self.class_prob {

            if probs.is_empty() {
                return Err(AnvilError::InvalidParam {
                    param: "class_prob",
                    reason: "cannot be empty".into(),
                });
            }

            let mut sum = 0.0;

            for &p in probs {
                if !(p >= 0.0 && p.is_finite()) {
                    return Err(AnvilError::InvalidParam {
                        param: "class_prob",
                        reason: "must contain finite, non-negative values".into(),
                    });
                }
                sum += p;
            }

            if sum <= 0.0 {
                return Err(AnvilError::InvalidParam {
                    param: "class_prob",
                    reason: "sum must be > 0".into(),
                });
            }
        }

        Ok(MultinomialNB {
            log_prob: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
        })
    }
}

impl MultinomialNB {
    /// Returns a new instance of [`MultinomialNB`] with default settings.
    pub fn new() -> Result<Self, AnvilError> {
        Builder::default().build()
    }

    /// Returns a [`Builder`] to configure the model.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns the unique class labels found during fitting.
    pub fn classes(&self) -> &Vec<usize> {
        &self.classes
    }
}

impl Estimator<usize> for MultinomialNB {
    /// Fits the Multinomial Naive Bayes model to the training data $(X, y)$.
    ///
    /// # Errors
    ///
    /// * `AnvilError::DimensionMismatch`: If `x` and `y` have different sample counts.
    /// * `AnvilError::EmptyDataset`: If the input `x` contains no samples.
    /// * `AnvilError::InvalidParam`: If `x` contains negative values or if provided 
    ///   priors do not match the number of classes.
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

        if n_samples == 0 {
            return Err(AnvilError::EmptyDataset { target: "X" });
        }

        // Multinomial constraint: no negative values
        if x.iter().any(|&v| v < 0.0) {
            return Err(AnvilError::InvalidParam {
                param: "X",
                reason: "MultinomialNB requires non-negative features".into(),
            });
        }

        // Encode labels
        let mut encoder = LabelEncoder::new();
        let target = encoder.fit_transform(y)?;

        self.classes = encoder.classes()?.to_vec();
        let n_classes = self.classes.len();

        if let Some(priors) = &self.class_prob {
            if priors.len() != n_classes {
                return Err(AnvilError::InvalidParam {
                    param: "class_prob",
                    reason: "length must match number of classes".into(),
                });
            }
        }

        // Accumulate counts
        let mut sum = Array2::<f64>::zeros((n_classes, n_features));
        let mut total_words = vec![0.0; n_classes];
        let mut class_count = vec![0usize; n_classes];

        for (i, row) in x.outer_iter().enumerate() {
            let c = target[i];
            class_count[c] += 1;

            Zip::from(sum.row_mut(c))
                .and(&row)
                .for_each(|s, &val| *s += val);

            total_words[c] += row.sum();
        }

        // Compute log probabilities (Laplace smoothing)
        let mut log_prob = Array2::<f64>::zeros((n_classes, n_features));

        for c in 0..n_classes {
            let denom = total_words[c] + n_features as f64;

            Zip::from(log_prob.row_mut(c))
                .and(sum.row(c))
                .for_each(|lp, &s| {
                    *lp = ((s + 1.0) / denom).ln();
                });
        }

        // Priors
        let total = n_samples as f64;

        let log_priors = if let Some(ref p) = self.class_prob {
            p.iter().map(|p| p.ln()).collect()
        } else {
            class_count
                .iter()
                .map(|&c| (c as f64 / total).ln())
                .collect()
        };

        self.log_prob = Some(log_prob);
        self.class_log_prior = Some(log_priors);

        Ok(())
    }
}

impl Classifier for MultinomialNB {
    /// Predicts class labels for a set of test samples.
    ///
    /// # Errors
    ///
    /// * `AnvilError::NotFitted`: If the model has not been trained.
    /// * `AnvilError::ShapeMismatch`: If the number of features in `x` differs 
    ///   from the training data.
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<usize>, AnvilError> {

        let log_prob = self.log_prob.as_ref().ok_or(AnvilError::NotFitted)?;
        let log_prior = self.class_log_prior.as_ref().ok_or(AnvilError::NotFitted)?;

        if x.ncols() != log_prob.ncols() {
            return Err(AnvilError::ShapeMismatch {
                expected: log_prob.ncols(),
                got: x.ncols(),
                axis: "features",
            });
        }

        let n_classes = log_prob.nrows();
        let mut preds = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                let mut score = log_prior[c];

                Zip::from(&row)
                    .and(log_prob.row(c))
                    .for_each(|&val, &lp| {
                        score += val * lp;
                    });

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            preds[i] = self.classes[best_class];
        }

        Ok(preds)
    }
}
