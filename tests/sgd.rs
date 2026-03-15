use anvil::linear_model::SGDRegressor;
use ndarray::array;

#[test]
fn fit_runs() {

    let x = array![
        [1.0,2.0],
        [2.0,1.0],
        [3.0,3.0],
        [4.0,4.0]
    ];

    let y = array![3.0,3.0,6.0,8.0];

    let mut model = SGDRegressor::new();

    model.fit(x.view(), y.view());
}

#[test]
fn predict_shape() {

    let x = array![
        [1.0,1.0],
        [2.0,2.0],
        [3.0,3.0],
        [4.0,4.0]
    ];

    let y = array![2.0,4.0,6.0,8.0];

    let mut model = SGDRegressor::new();

    model.fit(x.view(), y.view());

    let preds = model.predict(x.view());

    assert_eq!(preds.len(), x.nrows());
}

#[test]
fn learns_linear_relation() {

    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ];

    let y = array![2.0,4.0,6.0,8.0,10.0];

    let mut model = SGDRegressor::builder()
        .epochs(500)
        .build();

    model.fit(x.view(), y.view());

    let preds = model.predict(x.view());

    let mse: f64 = preds.iter()
        .zip(y.iter())
        .map(|(p,t)| (p - t).powi(2))
        .sum::<f64>() / y.len() as f64;

    assert!(mse < 1.0);
}

#[test]
#[should_panic]
fn dimension_mismatch() {

    let x = array![
        [1.0,1.0],
        [2.0,2.0]
    ];

    let y = array![1.0];

    let mut model = SGDRegressor::new();

    model.fit(x.view(), y.view());
}

#[test]
fn weights_change_after_training() {

    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ];

    let y = array![2.0,4.0,6.0,8.0];

    let mut model = SGDRegressor::builder()
        .epochs(200)
        .build();

    model.fit(x.view(), y.view());

    let weights = model.weights();

    let norm: f64 = weights.iter().map(|w| w.abs()).sum();

    assert!(norm > 0.0);
}

