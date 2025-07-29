# Neural Network Decision Boundary Analysis with Keras and Synthetic Data

This project explores how **initial weight configurations**, **dataset types**, and **network architectures** affect the **learning performance and decision boundaries** of a simple neural network built using TensorFlow/Keras.

We experiment on two popular synthetic datasets: `make_moons` (binary classification) and `make_classification` (multi-class). The results are visualized using decision boundaries to better understand how the model separates data classes.

---

## 📊 Project Objectives

- Compare different weight initialization strategies (zeros, constant, random, He, Glorot).
- Visualize decision boundaries of trained neural networks.
- Demonstrate effects of poor vs. proper initialization.
- Test the network on both binary and multi-class classification problems.

---

## 📁 Project Structure


---

## 🧪 Experiments

### 1. **Binary Classification with `make_moons`**

- A 2D dataset with non-linear separation.
- Tried 3 weight initializations:
  - **All Zeros** → Failed to learn (loss stuck).
  - **Constant (0.5)** → Minor improvement, still poor.
  - **Random scaled normal** → Successful learning.

### 2. **Multi-Class Classification with `make_classification`**

- 3-class 2D dataset with informative features.
- Used proper weight initialization:
  - **He normal** for ReLU activation.
  - **Glorot uniform** for softmax output.
- Achieved good accuracy and smooth decision boundaries.

---

## 📷 Visualization

We use `mlxtend.plotting.plot_decision_regions()` to visualize how the model classifies input space. This allows for intuitive comparison between different initialization methods.

---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow mlxtend
