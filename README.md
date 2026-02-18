# 🔨 anvil

A Rust-based machine learning library built from scratch with a focus on clarity, performance, and clean architecture.

---

## ✨ Highlights

- No external ML dependencies (as of now)
- Lightweight and fast
- Simple and intuitive API
- Designed for extensibility
- Built with performance-oriented Rust

---

## 🚀 Example

```rust
use anvil::linear_model::LogisticRegression;

let model = LogisticRegression::builder()
    .epochs(500)
    .batch_size(32)
    .build();

model.fit(&features, &targets);

let prediction = model.predict(&[2.0, 3.0]);
```
---

## 📌 Status

Early-stage development.  
Actively evolving.
