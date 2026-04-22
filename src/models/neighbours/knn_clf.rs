use std::collections::HashMap;

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{
    core::{Estimator, Classifier, AnvilError},
    neighbours::{NeighbourSearch, Weight},
    neighbours::metric::Euclidean,
    neighbours::brute_force::BruteForce,
};

pub struct KNNClassifier<N>
where
    N: NeighbourSearch,
{
    k: usize,
    searcher: N,
    weights: Weight,
    targets: Option<Vec<usize>>,
}

pub struct Builder<N>
where
    N: NeighbourSearch,
{
    k: usize,
    searcher: N,
    weights: Weight,
}

impl Default for Builder<BruteForce<Euclidean>> {
    fn default() -> Self {
        Self {
            k: 5,
            searcher: BruteForce::new(),
            weights: Weight::Uniform,
        }
    }
}


impl<N> Builder<N>
where
    N: NeighbourSearch,
{
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn weights(mut self, weights: Weight) -> Self {
        self.weights = weights;
        self
    }

    pub fn algorithm<S: NeighbourSearch>(self, searcher: S) -> Builder<S> {
        Builder {
            k: self.k,
            searcher,
            weights: self.weights,
        }
    }

    pub fn build(self) -> Result<KNNClassifier<N>,AnvilError> {

        if self.k == 0 {
            return Err(AnvilError::InvalidParam {
                param: "k",
                reason: "must be > 0".into(),
            });
        }

        Ok(KNNClassifier {
            k: self.k,
            searcher: self.searcher,
            weights: self.weights,
            targets: None,
        })
    }
}


impl KNNClassifier<BruteForce<Euclidean>> {
    pub fn new() -> Result<Self,AnvilError> {
        Builder::default().build()
    }

    pub fn builder() -> Builder<BruteForce<Euclidean>> {
        Builder::default()
    }
}

impl<N> Estimator<usize> for KNNClassifier<N>
where
    N: NeighbourSearch,
{
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<usize>,
    ) -> Result<(), AnvilError> {

        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }

        if n_samples == 0 {
            return Err(AnvilError::EmptyDataset { target: "X" });
        }

        if self.k == 0 {
            return Err(AnvilError::InvalidParam {
                param: "k",
                reason: "k must be > 0".into(),
            });
        }

        if self.k > n_samples {
            return Err(AnvilError::InvalidParam {
                param: "k",
                reason: "k cannot be greater than number of samples".into(),
            });
        }

        self.targets = Some(y.to_vec());
        self.searcher.build(x.to_owned());

        Ok(())
    }
}

impl<N> Classifier for KNNClassifier<N>
where
    N: NeighbourSearch,
{
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<usize>, AnvilError> {

        let targets = self.targets.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut predictions = Array1::zeros(x.nrows());
        let mut votes: HashMap<usize, f64> = HashMap::new();

        for (i, row) in x.outer_iter().enumerate() {

            let neighbours = self.searcher.query(row, self.k);

            votes.clear();

            for (idx, dist) in neighbours {

                let weight = match self.weights {
                    Weight::Uniform => 1.0,
                    Weight::Distance => {
                        if dist == 0.0 { 1.0 } else { 1.0 / dist }
                    }
                };

                *votes.entry(targets[idx]).or_insert(0.0) += weight;
            }

            let best = votes
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .ok_or(AnvilError::InvalidParam {
                    param: "knn",
                    reason: "no neighbours found".into(),
                })?;

            predictions[i] = *best.0;
        }

        Ok(predictions)
    }
}
