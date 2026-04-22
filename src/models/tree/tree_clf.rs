use std::cmp::Ordering;
use rand::{Rng, seq::SliceRandom};
use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{
    core::{Estimator, Classifier, AnvilError, Transformer},
    models::tree::{Criteria, Node},
    models::tree::impurity::{entropy, gini},
    preprocessing::encoder::LabelEncoder,
};

/// Decision Tree classifier.
///
/// Builds a binary tree by recursively splitting the feature space
/// using the best split found at each node.
///
/// # Notes
///
/// - Supports `Gini` and `Entropy` criteria
/// - Uses presorted indices — O(n log n) once at `fit`, O(n) per node
pub struct DecisionTreeClassifier {
    /// Impurity criterion used for split evaluation
    criteria: Criteria,

    /// Minimum number of samples required to split a node
    min_samples_split: usize,

    /// Minimum number of samples required in each leaf
    min_samples_leaf: usize,

    /// Maximum depth of the tree
    max_depth: Option<usize>,

    /// Max number of features to consider per split
    max_features: Option<usize>,

    /// Root node of the fitted tree
    root: Option<Box<Node<usize>>>,

    /// Original class labels.
    ///
    /// Maps encoded indices back to original labels during prediction.
    classes: Vec<usize>,
}

/// Builder for configuring [`DecisionTreeClassifier`]
pub struct Builder {
    criteria: Criteria,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - criteria = Gini
    /// - min_samples_split = 2
    /// - min_samples_leaf = 1
    /// - max_depth = None
    /// - max_features = None (uses all features)
    fn default() -> Self {
        Self {
            criteria: Criteria::Gini,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_depth: None,
            max_features: None,
        }
    }
}

impl Builder {
    /// Set impurity criterion
    pub fn criteria(mut self, criteria: Criteria) -> Self {
        self.criteria = criteria;
        self
    }

    /// Set minimum samples required to split a node
    pub fn min_samples_split(mut self, value: usize) -> Self {
        self.min_samples_split = value;
        self
    }

    /// Set minimum samples required in each leaf
    pub fn min_samples_leaf(mut self, value: usize) -> Self {
        self.min_samples_leaf = value;
        self
    }

    /// Set maximum tree depth
    pub fn max_depth(mut self, value: usize) -> Self {
        self.max_depth = Some(value);
        self
    }

    /// Set maximum number of features considered per split
    pub fn max_features(mut self, value: usize) -> Self {
        self.max_features = Some(value);
        self
    }

    /// Build model
    ///
    /// # Errors
    ///
    /// Returns `AnvilError::InvalidParam` if:
    ///
    /// - `min_samples_split` < 2
    /// - `min_samples_leaf` < 1
    /// - `max_features` < 1
    /// - `criteria` is not `Gini` or `Entropy`
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

        if let Some(max_features) = self.max_features {
            if max_features < 1 {
                return Err(AnvilError::InvalidParam {
                    param: "max_features",
                    reason: "must be >= 1".into(),
                });
            }
        }

        match self.criteria {
            Criteria::Gini | Criteria::Entropy => {}
            _ => {
                return Err(AnvilError::InvalidParam {
                    param: "criteria",
                    reason: "DecisionTreeClassifier only supports Gini or Entropy".into(),
                });
            }
        }

        Ok(DecisionTreeClassifier {
            criteria: self.criteria,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_depth: self.max_depth,
            max_features: self.max_features,
            root: None,
            classes: Vec::new(),
        })
    }
}

/// Shared context passed to split finding functions
struct SplitContext<'a> {
    features: ArrayView2<'a, f64>,
    target: ArrayView1<'a, usize>,
    sorted_idx: &'a [Vec<usize>],
    parent_counts: &'a [usize],
    parent_impurity: f64,
    n_samples: usize,
    n_classes: usize,
    min_samples_leaf: usize,
    criteria: fn(&[usize], usize) -> f64,
}

/// Result of a split search
struct SplitResult {
    best_ig: f64,
    best_feature: usize,
    best_threshold: f64,
    best_split_pos: usize,
}

/// Evaluates all split candidates for a single feature
#[inline]
fn evaluate_feature(ctx: &SplitContext, f: usize) -> (f64, usize, f64) {
    let col = &ctx.sorted_idx[f];

    let mut left_counts = vec![0usize; ctx.n_classes];
    let mut right_counts = ctx.parent_counts.to_vec();

    let mut best_ig = 0.0;
    let mut best_pos = 0;
    let mut best_threshold = 0.0;

    for j in 0..col.len() - 1 {
        let curr = col[j];
        let next = col[j + 1];

        left_counts[ctx.target[curr]] += 1;
        right_counts[ctx.target[curr]] -= 1;

        let left_size = j + 1;
        let right_size = ctx.n_samples - left_size;

        if left_size < ctx.min_samples_leaf || right_size < ctx.min_samples_leaf {
            continue;
        }

        // skip if same feature value — not a valid split boundary
        if ctx.features[[curr, f]] == ctx.features[[next, f]] {
            continue;
        }

        let ig = ctx.parent_impurity
            - (left_size as f64 / ctx.n_samples as f64)
                * (ctx.criteria)(&left_counts, left_size)
            - (right_size as f64 / ctx.n_samples as f64)
                * (ctx.criteria)(&right_counts, right_size);

        if ig > best_ig {
            best_ig = ig;
            best_pos = j;
            best_threshold =
                (ctx.features[[curr, f]] + ctx.features[[next, f]]) / 2.0;
        }
    }

    (best_ig, best_pos, best_threshold)
}

/// Finds the best split across a randomly shuffled subset of features.
///
/// When `max_features == n_features`, all features are evaluated in random order.
/// When `max_features < n_features`, only a random subset is evaluated.
fn find_best_split(
    ctx: &SplitContext,
    max_features: usize,
    rng: &mut impl Rng,
) -> SplitResult {
    let mut order: Vec<usize> = (0..ctx.sorted_idx.len()).collect();
    order.shuffle(rng);

    let mut best_ig = 0.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_split_pos = 0;

    for &f in order.iter().take(max_features) {
        let (ig, pos, threshold) = evaluate_feature(ctx, f);
        if ig > best_ig {
            best_ig = ig;
            best_feature = f;
            best_split_pos = pos;
            best_threshold = threshold;
        }
    }

    SplitResult {
        best_ig,
        best_feature,
        best_threshold,
        best_split_pos,
    }
}

/// Returns the majority class label among the given indices
#[inline]
fn majority(indices: &[usize], target: ArrayView1<usize>, n_classes: usize) -> usize {
    let mut counts = vec![0usize; n_classes];

    for &i in indices {
        counts[target[i]] += 1;
    }

    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| c)
        .map(|(i, _)| i)
        .unwrap()
}

/// Traverses the tree for a single sample and returns the encoded class index
#[inline]
fn traverse(node: &Node<usize>, row: ArrayView1<f64>) -> usize {
    match node {
        Node::Leaf { value } => *value,
        Node::Internal {
            feature,
            threshold,
            left,
            right,
        } => {
            if row[*feature] <= *threshold {
                traverse(left, row)
            } else {
                traverse(right, row)
            }
        }
    }
}

impl DecisionTreeClassifier {
    /// Create model with default configuration
    ///
    /// # Errors
    ///
    /// Returns `AnvilError::InvalidParam` if default builder validation fails
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

impl Estimator<usize> for DecisionTreeClassifier {
    /// Fits the decision tree classifier
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `features` and `target` row count differ
    /// - `InvalidParam` if dataset contains zero samples
    /// - `InvalidParam` if `max_features` > `n_features`
    fn fit(
        &mut self,
        features: ArrayView2<f64>,
        target: ArrayView1<usize>,
    ) -> Result<(), AnvilError> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        let max_features = self.max_features.unwrap_or(n_features);

        if n_samples != target.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: target.len(),
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
                reason: "cannot be greater than total number of features".into(),
            });
        }

        // encode class labels into contiguous indices
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target)?;

        let n_classes = encoder.classes()?.len();
        self.classes = encoder.classes()?.to_vec();

        // precompute sorted indices per feature — O(n log n) once
        let sorted_idx: Vec<Vec<usize>> = (0..n_features)
            .map(|f| {
                let mut col: Vec<usize> = (0..n_samples).collect();
                col.sort_unstable_by(|&a, &b| {
                    features[[a, f]]
                        .partial_cmp(&features[[b, f]])
                        .unwrap_or(Ordering::Equal)
                });
                col
            })
            .collect();

        // select impurity function at fit time — no branching in inner loop
        let impurity_fn: fn(&[usize], usize) -> f64 = match self.criteria {
            Criteria::Entropy => entropy,
            Criteria::Gini => gini,
            _ => {
                return Err(AnvilError::InvalidParam {
                    param: "criteria",
                    reason: "only Gini or Entropy supported for classification".into(),
                });
            }
        };

        let mut rng = rand::rng();

        self.root = Some(Box::new(self.build_tree(
            sorted_idx,
            0,
            features,
            ArrayView1::from(&encoded),
            n_classes,
            max_features,
            impurity_fn,
            &mut rng,
        )));

        Ok(())
    }
}

impl Classifier for DecisionTreeClassifier {
    /// Predict target labels
    ///
    /// # Errors
    ///
    /// - `NotFitted` if model has not been trained
    /// - `ShapeMismatch` if feature count does not match training data
    fn predict(&self, features: ArrayView2<f64>) -> Result<Array1<usize>, AnvilError> {
        let root = self.root.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {
            let encoded = traverse(root, row);
            preds[i] = self.classes[encoded];
        }

        Ok(preds)
    }
}

impl DecisionTreeClassifier {
    fn build_tree(
        &self,
        sorted_idx: Vec<Vec<usize>>,
        depth: usize,
        features: ArrayView2<f64>,
        target: ArrayView1<usize>,
        n_classes: usize,
        max_features: usize,
        criteria: fn(&[usize], usize) -> f64,
        rng: &mut impl Rng,
    ) -> Node<usize> {
        let n_samples = sorted_idx[0].len();
        let first_col = &sorted_idx[0];

        // 1. small node
        if n_samples < self.min_samples_split {
            return Node::Leaf {
                value: majority(first_col, target, n_classes),
            };
        }

        // 2. pure node
        let first = target[first_col[0]];
        if first_col.iter().all(|&i| target[i] == first) {
            return Node::Leaf { value: first };
        }

        // 3. max depth
        if let Some(max_d) = self.max_depth {
            if depth >= max_d {
                return Node::Leaf {
                    value: majority(first_col, target, n_classes),
                };
            }
        }

        // 4. parent impurity
        let mut parent_counts = vec![0usize; n_classes];
        for &i in first_col {
            parent_counts[target[i]] += 1;
        }

        let parent_impurity = criteria(&parent_counts, n_samples);

        let ctx = SplitContext {
            features,
            target,
            sorted_idx: &sorted_idx,
            parent_counts: &parent_counts,
            parent_impurity,
            n_samples,
            n_classes,
            min_samples_leaf: self.min_samples_leaf,
            criteria,
        };

        // 5. find best split
        let result = find_best_split(&ctx, max_features, rng);

        // 6. no valid split found
        if result.best_ig == 0.0 {
            return Node::Leaf {
                value: majority(first_col, target, n_classes),
            };
        }

        // 7. partition — mark left indices using best split
        let mut mark = vec![false; features.nrows()];
        for &idx in &sorted_idx[result.best_feature][..=result.best_split_pos] {
            mark[idx] = true;
        }

        let left_cap = result.best_split_pos + 1;
        let right_cap = n_samples - left_cap;
        let n_features = sorted_idx.len();

        let mut left_sorted: Vec<Vec<usize>> = (0..n_features)
            .map(|_| Vec::with_capacity(left_cap))
            .collect();

        let mut right_sorted: Vec<Vec<usize>> = (0..n_features)
            .map(|_| Vec::with_capacity(right_cap))
            .collect();

        for (f, col) in sorted_idx.iter().enumerate() {
            for &idx in col {
                if mark[idx] {
                    left_sorted[f].push(idx);
                } else {
                    right_sorted[f].push(idx);
                }
            }
        }

        // reset mark — avoid reallocation
        for &idx in &sorted_idx[result.best_feature][..=result.best_split_pos] {
            mark[idx] = false;
        }

        // 8. recurse
        let left = self.build_tree(
            left_sorted,
            depth + 1,
            features,
            target,
            n_classes,
            max_features,
            criteria,
            rng,
        );

        let right = self.build_tree(
            right_sorted,
            depth + 1,
            features,
            target,
            n_classes,
            max_features,
            criteria,
            rng,
        );

        Node::Internal {
            feature: result.best_feature,
            threshold: result.best_threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}
