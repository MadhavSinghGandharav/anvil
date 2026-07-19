# Anvil

A machine learning library written in Rust, built for performance and ergonomics.

Anvil provides classical ML algorithms with a consistent builder API across all models. It uses [`ndarray`](https://github.com/rust-ndarray/ndarray) for numerical operations and is designed to be zero-cost where possible — generics over dynamic dispatch, compile-time algorithm selection, and SIMD-friendly inner loops throughout.

---

## Features

- **Linear models** — SGD regression, logistic regression, perceptron
- **Naive Bayes** — Gaussian, Bernoulli , Multinomial
- **Neighbours** — KNN classifier and regressor with pluggable search algorithms and distance metrics
- **Trees** - DecisionTree classifier and regressor
- **Preprocessing** - Scalers (StandardScaler,MinMaxScaler,MaxAbsolute,QuantileTransformer) and Encoders (LabelEncoder) 
- Consistent builder API across all models
- Pluggable optimizers via the `Optimizer` trait
- Mini-batch training with shuffled SGD

---

## Installation

```toml
[dependencies]
anvil = { path = "." }
ordered-float = "4"
ndarray = "0.16"
```

---

## Usage

All models follow the same pattern:

```rust
let mut model = Model::builder()
    .option(value)
    .build();

model.fit(x_train.view(), y_train.view());
let predictions = model.predict(x_test.view());
```

---

## Example:

```rust
// default — BruteForce + Euclidean
let mut model = KNNClassifier::new();

// KDTree with distance weighting
let mut model = KNNClassifier::builder()
    .k(5)
    .weights(Weight::Distance)
    .algorithm(KDTree::with_leaf_size(30))
    .build();

// BruteForce with custom metric
let mut model = KNNClassifier::builder()
    .algorithm(BruteForce::with_metric(Manhattan))
    .build();
```

---

## Contributors
<a href="https://github.com/Akhilx12/anvil/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Akhilx12/anvil" />
</a>



## License

MIT
