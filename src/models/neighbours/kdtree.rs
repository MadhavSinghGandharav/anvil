use std::collections::BinaryHeap;
use std::cmp::Ordering;
use ndarray::{Array2, ArrayView1, Zip};
use ordered_float::OrderedFloat;
use crate::models::neighbours::NeighbourSearch;

/// KD-Tree nearest neighbour search.
///
/// Recursively partitions the feature space along alternating
/// dimensions using median splits, enabling efficient pruning
/// during nearest neighbour queries.
///
/// # Complexity
///
/// ```text
/// Build : O(n log n)
/// Query : O(log n) average, O(n) worst case
/// ```
///
/// # Numerical Notes
///
/// Distances are computed and stored as **squared Euclidean** during
/// traversal to avoid repeated `sqrt` calls. The final result
/// converts to true Euclidean distance.
pub struct KDTree {

    /// Training feature matrix of shape `(n_samples, n_features)`
    features: Option<Array2<f64>>,

    /// Root node of the tree
    root: Option<Node>,

    /// Maximum number of points per leaf node
    leaf_size: usize,
}

/// Internal KD-Tree node
enum Node {

    /// Leaf node storing indices into the training matrix
    Leaf {
        indices: Vec<usize>,
    },

    /// Internal split node
    Internal {
        /// Dimension used for splitting
        split_dim: usize,

        /// Split value along `split_dim`
        split_val: f64,

        /// Index of the median element in the training matrix
        median_index: usize,

        /// Left subtree — points with value ≤ `split_val`
        left: Box<Node>,

        /// Right subtree — points with value > `split_val`
        right: Box<Node>,
    },
}

/// Recursively builds the KD-Tree by median-partitioning `indices`
fn build_tree(
    indices: &mut [usize],
    features: &Array2<f64>,
    depth: usize,
    leaf_size: usize,
) -> Node {

    if indices.len() <= leaf_size {
        return Node::Leaf {
            indices: indices.to_vec(),
        };
    }

    // cycle through dimensions at each depth level
    let split_dim = depth % features.ncols();
    let mid = indices.len() / 2;

    // partition around median — O(n) average via Quickselect
    let (left_slice, median, right_slice) =
        indices.select_nth_unstable_by(mid, |&i, &j| {
            features[[i, split_dim]]
                .partial_cmp(&features[[j, split_dim]])
                .unwrap_or(Ordering::Equal)
        });

    let median_index = *median;
    let split_val = features[[median_index, split_dim]];

    Node::Internal {
        split_dim,
        split_val,
        median_index,
        left: Box::new(build_tree(left_slice, features, depth + 1, leaf_size)),
        right: Box::new(build_tree(right_slice, features, depth + 1, leaf_size)),
    }
}

/// Computes squared Euclidean distance between two points
#[inline]
fn squared_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    Zip::from(a)
        .and(b)
        .fold(0.0, |acc, &x, &y| {
            let diff = x - y;
            acc + diff * diff
        })
}

/// Pushes `(dist, idx)` into the max-heap if it improves the current top-k
#[inline]
fn push_if_closer(
    heap: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    dist: OrderedFloat<f64>,
    idx: usize,
    k: usize,
) {
    if heap.len() < k {
        heap.push((dist, idx));
    } else if let Some(&(worst, _)) = heap.peek() {
        if dist < worst {
            heap.pop();
            heap.push((dist, idx));
        }
    }
}

/// Recursively traverses the KD-Tree to collect the `k` nearest neighbours
fn traverse_knn(
    node: &Node,
    features: &Array2<f64>,
    query: ArrayView1<f64>,
    k: usize,
    heap: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
) {
    match node {

        Node::Leaf { indices } => {
            for &i in indices {
                let dist = OrderedFloat(squared_distance(features.row(i), query));
                push_if_closer(heap, dist, i, k);
            }
        }

        Node::Internal { split_dim, split_val, median_index, left, right } => {

            // evaluate median point
            let dist = OrderedFloat(squared_distance(features.row(*median_index), query));
            push_if_closer(heap, dist, *median_index, k);

            let query_val = query[*split_dim];

            // traverse near branch first
            let (near, far) = if query_val <= *split_val {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            traverse_knn(near, features, query, k, heap);

            // axis-aligned pruning — skip far branch if it cannot
            // contain a point closer than the current worst neighbour
            let plane_dist = OrderedFloat((query_val - split_val).powi(2));
            let worst = heap
                .peek()
                .map(|&(d, _)| d)
                .unwrap_or(OrderedFloat(f64::INFINITY));

            if heap.len() < k || plane_dist < worst {
                traverse_knn(far, features, query, k, heap);
            }
        }
    }
}

impl KDTree {

    /// Create model with default leaf size (40)
    pub fn new() -> Self {
        Self {
            features: None,
            root: None,
            leaf_size: 40,
        }
    }

    /// Create model with custom leaf size
    pub fn with_leaf_size(leaf_size: usize) -> Self {
        Self {
            features: None,
            root: None,
            leaf_size,
        }
    }
}

impl NeighbourSearch for KDTree {

    /// Builds the KD-Tree from the training matrix
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - dataset contains zero samples
    fn build(&mut self, data: Array2<f64>) {

        assert!(data.nrows() > 0, "Cannot build tree with zero samples");

        // collect row indices and partition recursively
        let mut indices: Vec<usize> = (0..data.nrows()).collect();
        let root = build_tree(&mut indices, &data, 0, self.leaf_size);

        self.features = Some(data);
        self.root = Some(root);
    }

    /// Returns the `k` nearest neighbours for a given query point
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - model not fitted
    /// - `k` is zero
    /// - query point dimension does not match training data
    fn query(&self, point: ArrayView1<f64>, k: usize) -> Vec<(usize, f64)> {

        let features = self.features.as_ref().expect("Model not fitted");
        let root = self.root.as_ref().expect("Model not fitted");

        assert!(k > 0, "k must be greater than 0");

        assert_eq!(
            features.ncols(),
            point.len(),
            "Query point dimension mismatch"
        );

        let mut heap = BinaryHeap::with_capacity(k);

        // traverse tree and collect k nearest neighbours
        traverse_knn(root, features, point, k, &mut heap);

        // convert squared distances to true Euclidean and return
        heap.into_iter()
            .map(|(dist, idx)| (idx, dist.into_inner().sqrt()))
            .collect()
    }
}
