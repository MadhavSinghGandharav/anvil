use anvil::linear_model::LogisticRegression;
use ndarray::array;

#[test]
fn fit_runs() {
    let x = array![
        [1.0, 2.0],
        [2.0, 1.0],
        [-1.0, -2.0],
        [-2.0, -1.0]
    ];

    let y = array![1, 1, 0, 0];

    let mut model = LogisticRegression::new();
    model.fit(x.view(), y.view());
}

#[test]
fn predict_shape() {
    let x = array![
        [1.0,1.0],
        [2.0,2.0],
        [-1.0,-1.0],
        [-2.0,-2.0]
    ];

    let y = array![1,1,0,0];

    let mut model = LogisticRegression::new();
    model.fit(x.view(), y.view());

    let preds = model.predict(x.view());

    assert_eq!(preds.len(), x.nrows());
}

#[test]
fn learns_simple_dataset() {
    let x = array![
        [2.0,2.0],
        [3.0,3.0],
        [4.0,4.0],
        [-2.0,-2.0],
        [-3.0,-3.0],
        [-4.0,-4.0]
    ];

    let y = array![1,1,1,0,0,0];

    let mut model = LogisticRegression::builder()
        .epochs(500)
        .build();

    model.fit(x.view(), y.view());

    let preds = model.predict(x.view());

    let correct = preds
        .iter()
        .zip(y.iter())
        .filter(|(p,t)| p == t)
        .count();

    let acc = correct as f64 / y.len() as f64;

    assert!(acc > 0.9);
}

#[test]
fn probability_range() {
    let x = array![
        [1.0,1.0],
        [2.0,2.0],
        [-1.0,-1.0],
        [-2.0,-2.0]
    ];

    let y = array![1,1,0,0];

    let mut model = LogisticRegression::new();
    model.fit(x.view(), y.view());

    let probs = model.predict_proba(x.view());

    for p in probs {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

#[test]
#[should_panic]
fn dimension_mismatch() {
    let x = array![
        [1.0,1.0],
        [2.0,2.0]
    ];

    let y = array![1];

    let mut model = LogisticRegression::new();
    model.fit(x.view(), y.view());
}

#[test]
fn weights_change_after_training() {
    let x = array![
        [2.0,2.0],
        [3.0,3.0],
        [-2.0,-2.0],
        [-3.0,-3.0]
    ];

    let y = array![1,1,0,0];

    let mut model = LogisticRegression::builder()
        .epochs(200)
        .build();

    model.fit(x.view(), y.view());

    let weights = model.weights();

    let norm: f64 = weights.iter().map(|w| w.abs()).sum();

    assert!(norm > 0.0);
}
