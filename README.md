# Description

Repository of differentially-private methods for learning from features.

# Usage
## Installation
`pip install dp-learning-ff`


## Private Prototype Calculation
1. Set your total privacy budget in (0,rho)-zCDP.
2. Divide your total budget into a budget per step. Good starting values are listed below. The algorithm will perform `len(Ps)` many steps.

```python
Ps = [rho] # One iteration
Ps = [5 / 64 * rho, 7 / 64 * rho, 52 / 64 * rho] # Three iterations
```

3. Call the private prototype calculation using your `train_preds`, `train_targets` and `Ps` as outlined above.

```python
num_samples = 100
dimensionality = 10
num_classes = 5
train_preds = np.random.normal(0,1,(num_samples, dimensionality))
train_targets = np.random.randint(num_classes, size=(num_samples)) # Supports unbalanced classes
private_prototypes = dp_learning_ff.give_private_prototypes(train_preds, train_targets, Ps)
>>> private_prototypes.shape
(5, 10)
>>> private_prototypes
array([[-1.08239659,  0.50517265,  0.46295668,  0.11665974, -0.41539363,
         0.46428818, -0.30204127, -0.20684517, -0.11748276, -0.44173581],
       [-0.97640077,  0.73495051, -0.3125302 ,  0.39733846, -0.15187237,
        -0.09618042,  0.50008494,  0.06408969, -0.59472233,  0.56000984],
       [-0.07384877, -0.04661516,  0.83469973, -0.18854654, -0.3631222 ,
        -0.56158617,  0.03480437, -0.17749153, -0.25144926,  0.99669163],
       [ 0.02682651, -0.57948712, -0.33462215,  0.32657359, -0.13894844,
        -1.16420508,  0.84630674, -0.0343235 , -0.38193481,  0.29124748],
       [ 1.12158357,  0.17188541, -0.2582887 , -0.2140504 ,  0.06759121,
        -0.26975582, -0.46858189,  0.67295234,  0.64183255, -0.10445519]])
```