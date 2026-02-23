use std::collections::BinaryHeap;
use std::cmp::Ordering;

use ordered_float::OrderedFloat;

use crate::core::DenseMatrix;
use crate::neighbours::{NeighbourSearch};
use crate::neighbours::metrics::Euclidean;

/// KD-Tree based nearest neighbor search structure.
///
/// A space-partitioning tree optimized for efficient k-nearest
/// neighbor queries under **Euclidean (L2) distance**.
///
/// # Design
///
/// - The tree owns the training feature matrix.
/// - During [`fit`], the dataset is partitioned recursively along
///   alternating feature dimensions.
/// - Each internal node splits data by a median value.
/// - Leaf nodes store indices of points.
/// - Queries use recursive traversal with pruning.
///
/// # Distance Semantics
///
/// Internally, the tree computes **squared Euclidean distance**
/// for performance and pruning consistency.
///
/// However, the [`neighbours`] method returns the true
/// Euclidean distance (i.e., the square root is applied
/// before returning results).
///
/// # Performance
///
/// - Build time: `O(n log n)`
/// - Average query: `O(log n)`
/// - Worst-case query: `O(n)`
///
/// ⚠️ Performance degrades in high-dimensional spaces.
///
/// # Limitations
///
/// - Supports **Euclidean distance only**.
/// - Must call [`fit`] before calling [`neighbours`].
pub struct KDTree {
    /// Training feature matrix (set after `fit`)
    features: Option<DenseMatrix>,

    /// Root node of the tree
    root: Option<Node>,

    /// Maximum number of points per leaf
    leaf_size: usize,
}

/// Internal KD-Tree node.
enum Node {
    /// Leaf node storing indices of points.
    Leaf {
        indices: Vec<usize>,
    },

    /// Internal split node.
    Internal {
        /// Dimension used for splitting.
        split_dim: usize,

        /// Split value for the dimension.
        split_val: f64,

        /// Index of the median element.
        median_index: usize,

        /// Left subtree.
        left: Box<Node>,

        /// Right subtree.
        right: Box<Node>,
    },
}

impl KDTree {

    /// Creates a new KDTree.
    ///
    /// # Arguments
    ///
    /// * `leaf_size` — Maximum number of points stored in a leaf node.
    ///
    /// Smaller leaf sizes:
    /// - Increase tree depth
    /// - Reduce brute-force comparisons inside leaves
    ///
    /// Larger leaf sizes:
    /// - Reduce tree depth
    /// - Increase per-leaf scanning cost
    pub fn new(leaf_size: usize) -> Self {
        Self {
            features: None,
            root: None,
            leaf_size,
        }
    }

    /// Computes squared Euclidean distance between two vectors.
    ///
    /// This is used internally to:
    /// - Avoid unnecessary `sqrt()` in hot loops
    /// - Maintain pruning consistency
    fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }

        sum
    }

    /// Recursively builds the KDTree.
    ///
    /// Splits points by median along alternating dimensions.
    fn build(
        indices: &mut [usize],
        features: &DenseMatrix,
        depth: usize,
        leaf_size: usize,
    ) -> Node {

        let n = indices.len();

        if n <= leaf_size {
            return Node::Leaf {
                indices: indices.to_vec(),
            };
        }

        let split_dim = depth % features.n_cols();
        let mid = n / 2;

        let (left_slice, median, right_slice) =
            indices.select_nth_unstable_by(mid, |&i, &j| {
                features
                    .get(i, split_dim)
                    .partial_cmp(&features.get(j, split_dim))
                    .unwrap_or(Ordering::Equal)
            });

        let median_index = *median;
        let split_val = features.get(median_index, split_dim);

        Node::Internal {
            split_dim,
            split_val,
            median_index,
            left: Box::new(Self::build(
                left_slice,
                features,
                depth + 1,
                leaf_size,
            )),
            right: Box::new(Self::build(
                right_slice,
                features,
                depth + 1,
                leaf_size,
            )),
        }
    }

    /// Recursive k-nearest neighbor traversal.
    ///
    /// Uses:
    /// - Max-heap to maintain top `k` closest points
    /// - Axis-aligned pruning to skip unnecessary branches
    fn traverse_knn(
        &self,
        query: &[f64],
        node: &Node,
        features: &DenseMatrix,
        k: usize,
        heap: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    ) {
        match node {

            Node::Leaf { indices } => {
                for &i in indices {
                    let row = features.row(i);
                    let dist = OrderedFloat(Self::squared_distance(row, query));

                    if heap.len() < k {
                        heap.push((dist, i));
                    } else if let Some(&(worst, _)) = heap.peek() {
                        if dist < worst {
                            heap.pop();
                            heap.push((dist, i));
                        }
                    }
                }
            }

            Node::Internal {
                split_dim,
                split_val,
                median_index,
                left,
                right,
            } => {

                // Evaluate median point
                let row = features.row(*median_index);
                let dist = OrderedFloat(Self::squared_distance(row, query));

                if heap.len() < k {
                    heap.push((dist, *median_index));
                } else if let Some(&(worst, _)) = heap.peek() {
                    if dist < worst {
                        heap.pop();
                        heap.push((dist, *median_index));
                    }
                }

                let query_val = query[*split_dim];

                let (near, far) = if query_val <= *split_val {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                self.traverse_knn(query, near, features, k, heap);

                // Axis-based pruning
                let plane_dist = OrderedFloat((query_val - split_val).powi(2));

                let worst_dist = heap
                    .peek()
                    .map(|(d, _)| *d)
                    .unwrap_or(OrderedFloat(f64::INFINITY));

                if heap.len() < k || plane_dist < worst_dist {
                    self.traverse_knn(query, far, features, k, heap);
                }
            }
        }
    }
}

impl NeighbourSearch<Euclidean> for KDTree {

    /// Builds the KDTree from training data.
    ///
    /// Consumes the provided feature matrix and constructs
    /// the internal tree structure.
    fn fit(&mut self, features: DenseMatrix) {
        let mut indices: Vec<usize> =
            (0..features.n_rows()).collect();

        self.root = Some(Self::build(
            &mut indices,
            &features,
            0,
            self.leaf_size,
        ));

        self.features = Some(features);
    }

    /// Returns the `k` nearest neighbors under Euclidean distance.
    ///
    /// # Returns
    ///
    /// A vector of `(distance, index)` pairs.
    ///
    /// The returned distance is the **true Euclidean distance**
    /// (square root applied).
    ///
    /// # Panics
    ///
    /// Panics if `fit` has not been called.
    fn neighbours(
        &self,
        query: &[f64],
        k: usize,
        _metric: &Euclidean,
    ) -> Vec<(f64, usize)> {

        let features = self.features.as_ref().expect("model not fitted");

        assert_eq!(features.n_cols(), query.len());

        let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> =
            BinaryHeap::with_capacity(k);

        if let Some(root) = &self.root {
            self.traverse_knn(query, root, features, k, &mut heap);
        }

        heap.into_iter()
            .map(|(dist, idx)| (dist.into_inner().sqrt(), idx))
            .collect()
    }
}
