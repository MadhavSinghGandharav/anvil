use crate::core::{DenseMatrix,DenseVector};

fn pre_process(features: &DenseMatrix, target: &DenseVector) {
    let n_samples = features.n_rows();
    let n_features = features.n_cols();

    assert_eq!(n_samples, target.len());

    let mut feature_offset = vec![0.0; n_features];
    let mut target_offset = 0.0f64;

    for row in 0..n_samples {
        let row_slice = features.row(row);
        for col in 0..n_features {
            feature_offset[col] += row_slice[col];
        }
        target_offset += target.get(row);
    }
    
    target_offset /= n_samples as f64;
    for val in &mut feature_offset {
        *val /= n_samples as f64;
    }



}
