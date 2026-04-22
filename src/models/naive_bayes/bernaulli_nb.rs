//! Bernoulli Naive Bayes module.
//!
//! This module implements the Bernoulli Naive Bayes algorithm, which is suitable for 
//! discrete data. Unlike Multinomial NB, which uses word counts, Bernoulli NB 
//! is designed for binary/boolean features.

use crate::{
    preprocessing::encoder::LabelEncoder,
    core::{Estimator, Classifier, AnvilError,Transformer},
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

/// Bernoulli Naive Bayes classifier.
///
/// This classifier is suitable for discrete data where features are independent 
/// boolean variables. It binarizes input features based on a specified `threshold`.
///
/// # Examples
///
/// ```
/// use anvil::models::BernoulliNB;
///
/// let model = BernoulliNB::builder()
///     .threshold(0.5)
///     .build()
///     .unwrap();
/// ```
pub struct BernoulliNB {
    log_prob: Option<Array2<f64>>,
    log_prob_neg: Option<Array2<f64>>,
    class_log_prior: Option<Vec<f64>>,
    class_prob: Option<Vec<f64>>,
    classes: Vec<usize>,
    threshold: f64,
}

/// A builder for configuring and creating a [`BernoulliNB`] instance.
pub struct Builder {
    threshold: f64,
    class_prob: Option<Vec<f64>>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            class_prob: None,
        }
    }
}

impl Builder {
    /// Sets user-defined prior probabilities for the classes.
    ///
    /// If not provided, priors are calculated from the class frequencies in the training set.
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    /// Sets the threshold for binarizing (mapping to boolean) the input features.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Consumes the builder and returns a [`BernoulliNB`] instance.
    ///
    /// # Errors
    ///
    /// Returns [`AnvilError::InvalidParam`] if the threshold is not finite or if 
    /// `class_prob` is invalid.
    pub fn build(self) -> Result<BernoulliNB, AnvilError> {

        if !self.threshold.is_finite() {
            return Err(AnvilError::InvalidParam {
                param: "threshold",
                reason: "must be finite".into(),
            });
        }

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

        Ok(BernoulliNB {
            log_prob: None,
            log_prob_neg: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
            threshold: self.threshold,
        })
    }
}

impl BernoulliNB {
    /// Returns a new [`BernoulliNB`] with default parameters (threshold = 0.0).
    pub fn new() -> Result<Self, AnvilError> {
        Builder::default().build()
    }

    /// Returns a [`Builder`] to configure the model.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns the class labels learned during training.
    pub fn classes(&self) -> &Vec<usize> {
        &self.classes
    }
}

impl Estimator<usize> for BernoulliNB {
    /// Fits the Bernoulli Naive Bayes model to the training data $(X, y)$.
    ///
    /// Input features are binarized: values greater than `threshold` are treated as 1,
    /// and others as 0. Laplace smoothing (additive smoothing) is applied.
    ///
    /// # Errors
    ///
    /// * `AnvilError::DimensionMismatch`: If `x` and `y` have different sample counts.
    /// * `AnvilError::EmptyDataset`: If the input `x` contains no samples.
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

        let mut count = Array2::<f64>::zeros((n_classes, n_features));
        let mut class_count = vec![0usize; n_classes];

        // Perform binarization and count occurrences per class
        for (i, row) in x.outer_iter().enumerate() {
            let c = target[i];
            class_count[c] += 1;

            Zip::from(count.row_mut(c))
                .and(&row)
                .for_each(|cnt, &val| {
                    if val > self.threshold {
                        *cnt += 1.0;
                    }
                });
        }

        let mut log_prob = Array2::<f64>::zeros((n_classes, n_features));
        let mut log_prob_neg = Array2::<f64>::zeros((n_classes, n_features));

        // Compute log-probabilities with Laplace smoothing
        for c in 0..n_classes {
            let n = class_count[c] as f64 + 2.0;

            Zip::from(log_prob.row_mut(c))
                .and(log_prob_neg.row_mut(c))
                .and(count.row(c))
                .for_each(|lp, lp_neg, &cnt| {
                    let p = (cnt + 1.0) / n;
                    *lp = p.ln();
                    *lp_neg = (1.0 - p).ln();
                });
        }

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
        self.log_prob_neg = Some(log_prob_neg);
        self.class_log_prior = Some(log_priors);

        Ok(())
    }
}

impl Classifier for BernoulliNB {
    /// Predicts class labels for the provided test samples.
    ///
    /// # Errors
    ///
    /// * `AnvilError::NotFitted`: If the model has not been trained yet.
    /// * `AnvilError::ShapeMismatch`: If the feature count of `x` differs from the training data.
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<usize>, AnvilError> {

        let log_prob = self.log_prob.as_ref().ok_or(AnvilError::NotFitted)?;
        let log_prob_neg = self.log_prob_neg.as_ref().ok_or(AnvilError::NotFitted)?;
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
                    .and(log_prob_neg.row(c))
                    .for_each(|&val, &lp, &lp_neg| {
                        if val > self.threshold {
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

        Ok(preds)
    }
}
