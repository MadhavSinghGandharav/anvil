use std::cmp::Ordering;
use rand::Rng;
use ndarray::{ArrayView1, ArrayView2, Array1};

use crate::{
    core::{Estimator, AnvilError, Transformer, Classifier},
    models::trees::*,
    models::trees::tree_clf::*,
    preprocessing::encoder::LabelEncoder,
};

/// =======================
/// MODEL
/// =======================
pub struct DecisionTreeClassifier {
    criterion: Criterion,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
    root: Option<Box<Node<usize>>>,
    classes: Vec<usize>,

    sorted_idx: Vec<Vec<usize>>,
}

/// =======================
/// BUILDER
/// =======================
pub struct Builder {
    criterion: Criterion,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            criterion: Criterion::Gini,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_depth: None,
            max_features: None,
        }
    }
}

impl Builder {
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    pub fn min_samples_split(mut self, v: usize) -> Self {
        self.min_samples_split = v;
        self
    }

    pub fn min_samples_leaf(mut self, v: usize) -> Self {
        self.min_samples_leaf = v;
        self
    }

    pub fn max_depth(mut self, v: usize) -> Self {
        self.max_depth = Some(v);
        self
    }

    pub fn max_features(mut self, v: usize) -> Self {
        self.max_features = Some(v);
        self
    }

    pub fn build(self) -> Result<DecisionTreeClassifier, AnvilError> {
        if self.min_samples_split < 2 {
            return Err(AnvilError::InvalidParam {
                param: "min_samples_split",
                reason: "must be >= 2".into(),
            });
        }

        if self.min_samples_leaf < 1 {
            return Err(AnvilError::InvalidParam {
                param: "min_samples_leaf",
                reason: "must be >= 1".into(),
            });
        }

        if let Some(m) = self.max_features {
            if m < 1 {
                return Err(AnvilError::InvalidParam {
                    param: "max_features",
                    reason: "must be >= 1".into(),
                });
            }
        }

        Ok(DecisionTreeClassifier {
            criterion: self.criterion,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_depth: self.max_depth,
            max_features: self.max_features,
            root: None,
            classes: Vec::new(),
            sorted_idx: Vec::new(),
        })
    }
}

impl DecisionTreeClassifier {
    /// Create model with default configuration
    ///
    /// # Errors
    ///
    pub fn new() -> Result<Self, AnvilError> {
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
}

/// =======================
/// ESTIMATOR IMPL
/// =======================
impl Estimator<usize> for DecisionTreeClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) -> Result<(), AnvilError> {

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let max_features = self.max_features.unwrap_or(n_features);

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }

        if n_samples == 0 {
            return Err(AnvilError::InvalidParam {
                param: "features",
                reason: "cannot fit with zero samples".into(),
            });
        }

        if max_features > n_features {
            return Err(AnvilError::InvalidParam {
                param: "max_features",
                reason: "cannot be greater than total features".into(),
            });
        }

        // -------- Encode labels --------
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(y)?;

        let n_classes = encoder.classes()?.len();
        self.classes = encoder.classes()?.to_vec();

        // -------- Sorted indices --------
        let sorted_idx: Vec<Vec<usize>> = (0..n_features)
            .map(|f| {
                let mut col: Vec<usize> = (0..n_samples).collect();
                col.sort_unstable_by(|&a, &b| {
                    x[[a, f]]
                        .partial_cmp(&x[[b, f]])
                        .unwrap_or(Ordering::Equal)
                });
                col
            })
            .collect();

        self.sorted_idx = sorted_idx;

        // -------- Criterion --------
        let impurity_fn: fn(&[usize], usize) -> f64 = match self.criterion {
            Criterion::Entropy => entropy,
            Criterion::Gini => gini,
        };

        // -------- Root context --------
        let indices: Vec<usize> = (0..n_samples).collect();

        let ctx = SplitContext {
            x,
            y: encoded.view(),
            indices: &indices,
            depth: 0,
        };

        // -------- RNG --------
        let mut rng = rand::rng();

        // -------- Build tree --------
        let root = build_tree_clf(
            self,
            impurity_fn,
            &ctx,
            n_classes,
            self.min_samples_leaf,
            self.max_depth.unwrap_or(usize::MAX),
            self.min_samples_split,
            &mut rng,
        );

        self.root = Some(Box::new(root));

        Ok(())
    }
}


use rand::seq::SliceRandom;

impl ClfSplitter for DecisionTreeClassifier {
    fn best_split(
        &self,
        ctx: &SplitContext<usize>,
        criterion: fn(&[usize], usize) -> f64,
        n_classes: usize,
        rng: &mut impl Rng,
    ) -> Option<SplitResult> {

        let n_samples_total = ctx.x.nrows();
        let n_node_samples = ctx.indices.len();

        // ---------- mask ----------
        let mut in_node = vec![false; n_samples_total];
        for &i in ctx.indices {
            in_node[i] = true;
        }

        // ---------- parent counts ----------
        let mut parent_counts = vec![0usize; n_classes];
        for &i in ctx.indices {
            parent_counts[ctx.y[i]] += 1;
        }

        let mut best_feature = 0usize;
        let mut best_threshold = 0.0;
        let mut best_pos = 0usize;
        let mut best_impurity = f64::INFINITY;

        // ---------- feature selection ----------
        let mut features: Vec<usize> = (0..ctx.x.ncols()).collect();

        if let Some(max_f) = self.max_features {
            features.shuffle(rng);
            features.truncate(max_f);
        }

        // ---------- reusable buffers ----------
        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts_local = vec![0usize; n_classes];

        // ---------- iterate features ----------
        for &feature in &features {

            // reset buffers
            left_counts.fill(0);
            right_counts_local.clone_from(&parent_counts);

            let mut left_n = 0usize;
            let mut right_n = n_node_samples;

            let mut prev_value = None;
            let sorted = &self.sorted_idx[feature];

            for &idx in sorted {

                if !in_node[idx] {
                    continue;
                }

                let y_val = ctx.y[idx];

                // move right -> left
                left_counts[y_val] += 1;
                right_counts_local[y_val] -= 1;

                left_n += 1;
                right_n -= 1;

                let val = ctx.x[[idx, feature]];

                // skip identical values
                if let Some(prev) = prev_value {
                    if val == prev {
                        continue;
                    }
                }

                prev_value = Some(val);

                // leaf constraint
                if left_n < self.min_samples_leaf || right_n < self.min_samples_leaf {
                    continue;
                }

                // impurity
                let left_imp = criterion(&left_counts, left_n);
                let right_imp = criterion(&right_counts_local, right_n);

                let weighted = (left_n as f64 * left_imp
                    + right_n as f64 * right_imp)
                    / (n_node_samples as f64);

                if weighted < best_impurity {
                    best_impurity = weighted;
                    best_feature = feature;
                    best_threshold = val;
                    best_pos = left_n;
                }
            }
        }

        if best_impurity == f64::INFINITY {
            return None;
        }

        Some(SplitResult {
            feature: best_feature,
            threshold: best_threshold,
            pos: best_pos,
        })
    }
}

 impl Classifier for DecisionTreeClassifier{
     fn predict(&self, x: ArrayView2<f64>) -> Result<ndarray::Array1<usize>, AnvilError> {
          let root = self.root.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut preds = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {
            let encoded = traverse(root, row);
            preds[i] = self.classes[encoded];
        }

        Ok(preds)
    } 
 }
