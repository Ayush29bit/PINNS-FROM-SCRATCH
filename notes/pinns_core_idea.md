# The Core Idea Behind Physics-Informed Neural Networks (PINNs)

## Motivation

Traditional neural networks often require large amounts of data to generalize well. In many scientific and engineering applications, collecting large datasets can be expensive, difficult, or even impossible.

However, many real-world systems already obey known physical laws that are expressed through differential equations.

Examples include:

* Heat transfer
* Fluid dynamics
* Wave propagation
* Conservation laws

A standard neural network may fit the available data but still produce predictions that violate these physical laws.

Physics-Informed Neural Networks (PINNs) solve this problem by incorporating physical knowledge directly into the training process.

---

## PINNs as a Physics-Based Constraint

Traditional regularization techniques such as:

* L1 Regularization
* L2 Regularization
* Dropout

help prevent overfitting by constraining the model.

PINNs introduce a different kind of constraint:

> Instead of only penalizing complex models, PINNs penalize solutions that violate known physical laws.

Physics therefore acts as an additional source of supervision during training.

---

## The Main Idea

A standard neural network learns only from data:

```text
Input → Neural Network → Prediction
```

A PINN learns from both:

```text
Data
+
Physics Knowledge
      ↓
Neural Network
      ↓
Prediction
```

The network is encouraged to:

1. Fit the observed data.
2. Satisfy the governing differential equations.

---

## Embedding Physics into the Loss Function

PINNs do not usually modify the neural network architecture.

Instead, they modify the loss function.

Traditional training:

```python
loss = data_loss
```

PINN training:

```python
loss = data_loss + physics_loss
```

The network is therefore optimized to satisfy both objectives simultaneously.

---

## PINN Loss Function

A typical PINN loss function is:

[
Loss_{PINN}
===========

\frac{1}{N}
\sum_{j=1}^{N}
||f(x_j|\theta)-y_j||^2
+
\lambda
\frac{1}{M}
\sum_{i=1}^{M}
||g(x_i,f(x_i,\theta))||^2
]

The loss consists of two major components:

### 1. Data Loss

[
\frac{1}{N}
\sum_{j=1}^{N}
||f(x_j|\theta)-y_j||^2
]

This is simply the Mean Squared Error (MSE) between:

* Actual observations (y_j)
* Neural network predictions (f(x_j|\theta))

Its purpose is to ensure that the model fits the available data.

---

### 2. Physics Loss

[
\lambda
\frac{1}{M}
\sum_{i=1}^{M}
||g(x_i,f(x_i,\theta))||^2
]

This term measures how much the neural network violates the governing differential equation.

If the physical law is perfectly satisfied:

[
g(x_i,f(x_i,\theta)) = 0
]

and the physics loss becomes zero.

Larger violations result in larger penalties.

---

## What is (g(x,y))?

The function (g) represents the governing physical equation.

For example, consider the differential equation:

[
\frac{dy}{dx} + y = 0
]

It can be rewritten as:

[
g(x,y)=\frac{dy}{dx}+y
]

A perfect solution satisfies:


g(x,y)=0


everywhere in the domain.

During training, the PINN continuously checks whether its predictions satisfy this equation.

---

## Automatic Differentiation

To evaluate the physics equation, derivatives of the neural network output are required.

PINNs use Automatic Differentiation (Autograd) to compute these derivatives efficiently.

---

## Collocation Points

Physics loss is evaluated at special points called **collocation points**.

These are locations within the domain where the differential equation is checked.

Example:

```text
Domain: x ∈ [0,10]

Collocation Points:
0.5
1.7
3.2
6.8
9.4
```

At each collocation point, the model is asked:

> "Does the predicted solution satisfy the physical law here?"

Unlike training data, collocation points do not require observed labels.

They only require knowledge of the governing equation.

---

## Role of λ (Lambda)

The parameter λ controls the balance between:

* Data Loss
* Physics Loss

### Small λ

The model focuses primarily on fitting data.

### Large λ

The model focuses more strongly on satisfying physics.

Choosing an appropriate value for λ is important because both objectives must be balanced.

---

## Training Workflow

The PINN training process can be summarized as:

```text
Input x
      ↓
Neural Network
      ↓
Prediction y
      ↓

Compute Data Loss

Compute Physics Loss

Total Loss
=
Data Loss
+
λ × Physics Loss

      ↓

Backpropagation
      ↓

Update Weights
```

The optimization process is identical to standard neural network training.

The key difference is that the model is guided by both data and physical laws.

---

## Key Takeaways

* PINNs combine machine learning with physical knowledge.
* Physics is incorporated through the loss function.
* The loss contains both data loss and physics loss.
* Automatic differentiation is used to compute derivatives.
* Collocation points are used to enforce physical laws.
* PINNs are especially useful when data is limited but governing equations are known.
* The central goal of a PINN is to find solutions that fit data while remaining physically consistent.
