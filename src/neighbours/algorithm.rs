use crate::core::DenseMatrix;
use crate::neighbours::DistanceMetric;
use std::cmp::Ordering;

/// Brute-force neighbor search implementation.
///
/// This strategy computes the distance between the query point
/// and every row in the training matrix, then selects the
/// `k` smallest distances using partial selection.
///
/// Time Complexity:
/// - Distance computation: O(n)
/// - Selection (Quickselect): O(n) average
///
/// Overall average complexity: O(n)
pub(crate) struct BruteForce;

impl BruteForce {

    /// Returns the `k` nearest neighbors for a given query point.
    ///
    /// # Arguments
    /// * `train`  - Training feature matrix
    /// * `query`  - Single query feature slice
    /// * `k`      - Number of neighbors to retrieve
    /// * `metric` - Distance metric implementation
    ///
    /// # Returns
    /// A vector of `(distance, row_index)` pairs for the
    /// `k` closest rows in the training data.
    ///
    /// The returned neighbors are not guaranteed to be sorted.
    pub fn neighbours<M: DistanceMetric>(
        train: &DenseMatrix,
        query: &[f64],
        k: usize,
        metric: &M,
    ) -> Vec<(f64, usize)> {

        let mut distances: Vec<(f64, usize)> = (0..train.n_rows())
            .map(|row| {
                let row_slice = train.row(row);
                let dist = metric.distance(row_slice, query);
                (dist, row)
            })
            .collect();

        // Ensure k does not exceed dataset size
        let kth = k.min(distances.len());

        // Quickselect partitioning: places the k-th smallest element
        // in its correct position and partitions the vector.
        distances.select_nth_unstable_by(kth - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        // Keep only the k smallest elements
        distances.truncate(kth);

        distances
    }
}


pub(crate) struct KDTree {
    root: Option<Node>,
    leaf_size: usize,
}

enum Node {
    Leaf {
        indices: Vec<usize>,
    },
    Internal {
        split_dim: usize,
        split_val: f64,
        median_index: usize,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl KDTree {
    pub(crate) fn new(leaf_size: usize) -> Self {
        Self {
            root: None,
            leaf_size,
        }
    }

    pub(crate) fn build_tree(&mut self, features: &DenseMatrix) {
        let mut indices: Vec<usize> = (0..features.n_rows()).collect();
        self.root = Some(Self::build(
            &mut indices,
            features,
            0,
            self.leaf_size,
        ));
    }

    fn build(
        indices: &mut [usize],
        features: &DenseMatrix,
        depth: usize,
        leaf_size: usize,
    ) -> Node {
        let n = indices.len();

        // Leaf case
        if n <= leaf_size {
            return Node::Leaf {
                indices: indices.to_vec(),
            };
        }

        let split_dim = depth % features.n_cols();
        let mid = n / 2;

        // Partition indices around median
        let (left_slice, median, right_slice) =
            indices.select_nth_unstable_by(mid, |&i, &j| {
                features
                    .get(i, split_dim)
                    .partial_cmp(&features.get(j, split_dim))
                    .unwrap_or(Ordering::Equal)
            });

        let median_index = *median;
        let split_val = features.get(median_index, split_dim);

        let left_node = if !left_slice.is_empty() {
            Box::new(Self::build(
                left_slice,
                features,
                depth + 1,
                leaf_size,
            ))
        } else {
            Box::new(Node::Leaf { indices: vec![] })
        };

        let right_node = if !right_slice.is_empty() {
            Box::new(Self::build(
                right_slice,
                features,
                depth + 1,
                leaf_size,
            ))
        } else {
            Box::new(Node::Leaf { indices: vec![] })
        };

        Node::Internal {
            split_dim,
            split_val,
            median_index,
            left: left_node,
            right: right_node,
        }
    }
pub(crate) struct KDTree {
    root: Option<Node>,
    leaf_size: usize,
}

enum Node {
    Leaf {
        indices: Vec<usize>,
    },
    Internal {
        split_dim: usize,
        split_val: f64,
        median_index: usize,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl KDTree {
    pub(crate) fn new(leaf_size: usize) -> Self {
        Self {
            root: None,
            leaf_size,
        }
    }

    pub(crate) fn build_tree(&mut self, features: &DenseMatrix) {
        let mut indices: Vec<usize> = (0..features.n_rows()).collect();
        self.root = Some(Self::build(
            &mut indices,
            features,
            0,
            self.leaf_size,
        ));
    }

    fn build(
        indices: &mut [usize],
        features: &DenseMatrix,
        depth: usize,
        leaf_size: usize,
    ) -> Node {
        let n = indices.len();

        // Leaf case
        if n <= leaf_size {
            return Node::Leaf {
                indices: indices.to_vec(),
            };
        }

        let split_dim = depth % features.n_cols();
        let mid = n / 2;

        // Partition indices around median
        let (left_slice, median, right_slice) =
            indices.select_nth_unstable_by(mid, |&i, &j| {
                features
                    .get(i, split_dim)
                    .partial_cmp(&features.get(j, split_dim))
                    .unwrap_or(Ordering::Equal)
            });

        let median_index = *median;
        let split_val = features.get(median_index, split_dim);

        let left_node = if !left_slice.is_empty() {
            Box::new(Self::build(
                left_slice,
                features,
                depth + 1,
                leaf_size,
            ))
        } else {
            Box::new(Node::Leaf { indices: vec![] })
        };

        let right_node = if !right_slice.is_empty() {
            Box::new(Self::build(
                right_slice,
                features,
                depth + 1,
                leaf_size,
            ))
        } else {
            Box::new(Node::Leaf { indices: vec![] })
        };

        Node::Internal {
            split_dim,
            split_val,
            median_index,
            left: left_node,
            right: right_node,
        }
    }
}
}
