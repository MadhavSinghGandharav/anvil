/// Entropy impurity: -Σ p * ln(p)
///
/// Used for classification. Skips zero counts to avoid `0 * ln(0) = NaN`.
///
/// ```text
/// H = -Σ p_i * ln(p_i)
/// ```
#[inline]
pub(super) fn entropy(counts: &[usize], total: usize) -> f64 {

    if total == 0 {
        return 0.0;
    }

    let mut sum = 0.0;

    for &c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total as f64;
        sum -= p * p.ln();
    }

    sum
}

/// Gini impurity: 1 - Σ p²
///
/// Used for classification.
///
/// ```text
/// G = 1 - Σ p_i²
/// ```
#[inline]
pub(super) fn gini(counts: &[usize], total: usize) -> f64 {

    if total == 0 {
        return 0.0;
    }

    let mut sum = 1.0;

    for &c in counts {
        let p = c as f64 / total as f64;
        sum -= p * p;
    }

    sum
}

/// Mean Squared Error impurity
///
/// Used for regression.
///
/// ```text
/// MSE = (1/n) Σ (y_i - ȳ)²
///           = E[y²] - ȳ²
/// ```
#[inline]
pub(super) fn mse(sum: f64, sum_sq: f64, total: usize) -> f64 {

    if total == 0 {
        return 0.0;
    }

    let n    = total as f64;
    let mean = sum / n;

    sum_sq / n - mean * mean
}

/// Mean Absolute Error impurity
///
/// Used for regression.
///
/// ```text
/// MAE = (1/n) Σ |y_i - median|
/// ```
#[inline]
pub(super) fn mae(values: &[f64], total: usize) -> f64 {

    if total == 0 {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if total % 2 == 0 {
        (sorted[total / 2 - 1] + sorted[total / 2]) / 2.0
    } else {
        sorted[total / 2]
    };

    let n = total as f64;

    sorted.iter().map(|&x| (x - median).abs()).sum::<f64>() / n
}
