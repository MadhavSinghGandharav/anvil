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

