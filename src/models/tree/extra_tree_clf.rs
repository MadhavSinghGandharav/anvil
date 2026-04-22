use rand::{Rng, RngExt};
use rand::seq::SliceRandom;
use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{
    core::{Estimator, Classifier, AnvilError, Transformer},
    models::tree::{Criteria, Node},
    models::tree::impurity::{entropy, gini},
    preprocessing::encoder::LabelEncoder,
};

/// Extra Tree classifier.
///
/// Builds a binary tree by recursively splitting the feature space
/// using a random threshold for each candidate feature.
///
/// # Notes
///
/// - Supports `Gini` and `Entropy` criteria
/// - No sorting required — random threshold in [min, max] per feature
/// - Much faster per node than Decision Tree at the cost of higher bias
pub struct ExtraTreeClassifier {
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

/// Builder for configuring [`ExtraTreeClassifier`]
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
    pub fn build(self) -> Result<ExtraTreeClassifier, AnvilError> {
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
                    reason: "ExtraTreeClassifier only supports Gini or Entropy".into(),
                });
            }
        }

        Ok(ExtraTreeClassifier {
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

/// Shared context passed to split finding function
struct SplitContext<'a> {
    features: ArrayView2<'a, f64>,
    target: ArrayView1<'a, usize>,
    indices: &'a [usize],
    parent_counts: &'a [usize],
    parent_impurity: f64,
    n_samples: usize,
    n_classes: usize,
    min_samples_leaf: usize,
    criteria: fn(&[usize], usize) -> f64,
}

/// Result of a split search — includes partition
struct SplitResult {
    best_ig: f64,
    best_feature: usize,
    best_threshold: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

/// Finds the best random split across a shuffled subset of features.
///
/// For each candidate feature, a random threshold is drawn from [min, max].
/// The split with the highest information gain is selected.
/// Partition into left/right indices is done once at the end.
fn find_random_best_split(
    ctx: &SplitContext,
    max_features: usize,
    rng: &mut impl Rng,
) -> SplitResult {
    let n_features = ctx.features.ncols();

    let mut order: Vec<usize> = (0..n_features).collect();
    order.shuffle(rng);

    // reusable buffers — avoid allocation per feature
    let mut left_counts = vec![0usize; ctx.n_classes];
    let mut right_counts = vec![0usize; ctx.n_classes];

    let mut best_ig = f64::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for &f in order.iter().take(max_features) {
        // compute min/max for this feature over current indices
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &idx in ctx.indices {
            let v = ctx.features[[idx, f]];
            if v < min { min = v; }
            if v > max { max = v; }
        }

        // all values identical — no valid split
        if min == max {
            continue;
        }

        let thr = rng.random_range(min..max);

        // reset counts
        left_counts.fill(0);
        right_counts.copy_from_slice(ctx.parent_counts);

        let mut left_size = 0;

        for &idx in ctx.indices {
            if ctx.features[[idx, f]] <= thr {
                left_counts[ctx.target[idx]] += 1;
                right_counts[ctx.target[idx]] -= 1;
                left_size += 1;
            }
        }

        let right_size = ctx.n_samples - left_size;

        if left_size < ctx.min_samples_leaf || right_size < ctx.min_samples_leaf {
            continue;
        }

        let ig = ctx.parent_impurity
            - (left_size as f64 / ctx.n_samples as f64)
                * (ctx.criteria)(&left_counts, left_size)
            - (right_size as f64 / ctx.n_samples as f64)
                * (ctx.criteria)(&right_counts, right_size);

        if ig > best_ig {
            best_ig = ig;
            best_feature = f;
            best_threshold = thr;
        }
    }

    // partition once using best feature + threshold
    let mut left_indices = Vec::with_capacity(ctx.indices.len());
    let mut right_indices = Vec::with_capacity(ctx.indices.len());

    for &idx in ctx.indices {
        if ctx.features[[idx, best_feature]] <= best_threshold {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    SplitResult {
        best_ig,
        best_feature,
        best_threshold,
        left_indices,
        right_indices,
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

impl ExtraTreeClassifier {
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

impl Estimator<usize> for ExtraTreeClassifier {
    /// Fits the extra tree classifier
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

        let indices: Vec<usize> = (0..n_samples).collect();

        let mut rng = rand::rng();

        self.root = Some(Box::new(self.build_tree(
            indices,
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

impl Classifier for ExtraTreeClassifier {
    /// Predict target labels
    ///
    /// # Errors
    ///
    /// - `NotFitted` if model has not been trained
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

impl ExtraTreeClassifier {
    fn build_tree(
        &self,
        indices: Vec<usize>,
        depth: usize,
        features: ArrayView2<f64>,
        target: ArrayView1<usize>,
        n_classes: usize,
        max_features: usize,
        criteria: fn(&[usize], usize) -> f64,
        rng: &mut impl Rng,
    ) -> Node<usize> {
        let n_samples = indices.len();

        // 1. small node
        if n_samples < self.min_samples_split {
            return Node::Leaf {
                value: majority(&indices, target, n_classes),
            };
        }

        // 2. pure node
        let first = target[indices[0]];
        if indices.iter().all(|&i| target[i] == first) {
            return Node::Leaf { value: first };
        }

        // 3. max depth
        if let Some(max_d) = self.max_depth {
            if depth >= max_d {
                return Node::Leaf {
                    value: majority(&indices, target, n_classes),
                };
            }
        }

        // 4. parent impurity
        let mut parent_counts = vec![0usize; n_classes];
        for &i in &indices {
            parent_counts[target[i]] += 1;
        }

        let parent_impurity = criteria(&parent_counts, n_samples);

        let ctx = SplitContext {
            features,
            target,
            indices: &indices,
            parent_counts: &parent_counts,
            parent_impurity,
            n_samples,
            n_classes,
            min_samples_leaf: self.min_samples_leaf,
            criteria,
        };

        // 5. find random best split
        let result = find_random_best_split(&ctx, max_features, rng);

        // 6. no valid split found
        if result.best_ig <= 0.0 {
            return Node::Leaf {
                value: majority(&indices, target, n_classes),
            };
        }

        // 7. recurse — partition already done inside find_random_best_split
        let left = self.build_tree(
            result.left_indices,
            depth + 1,
            features,
            target,
            n_classes,
            max_features,
            criteria,
            rng,
        );

        let right = self.build_tree(
            result.right_indices,
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
