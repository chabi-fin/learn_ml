---
share: true
---
Importance of gradients and activations and their statistics.

### Initialization
If the initial predictions are not chosen carefully, then the untrained network has far worse predictions than even random predictions. So, it's important to initialize with reasonable guesses. The initial loss will be capped near a lower value ($-\log(1/27)=3.3$) if the outputs are initialized with near zeros. This way, we avoid confidently wrong initial predictions. 

Importantly, deeper networks become more sensitive to bad initializations, so its important to be careful and to consider the ways training can run afoul.

Diagnosis : If a plot of the losses during training vs. training iterations resembles a hockey stick, this signifies that the initialized values should be corrected. 

Note that biases can be set to exactly zero, but weights should not be exactly zero. Output weights should be near zero, so that the gradient will still impact parameter updates. There is a specific strategy for deciding how close to zero to go: 

**Scale the input weights** : the large increase in the standard deviation after activation. This is undesirable. So the idea is to scale w, so that the standard deviation is roughly similar after activation. The heuristic is, divide the matrix by th square root of the number of inputs. So in this case above, $$y = x @ (w / \sqrt{10})$$In fact, torch has a commonly used function for initializing weights in this way : `torch.nn.init.kaiming_normal_()` (Kaiming He et al. authored the paper which analyses the weight initializations forward and back propagation behavior)
If the inputs are distributed mostly above $1$ of below $-1$, then it becomes very difficult to change parameter. The gradient update is `self.grad += (1 - h**2) * out.grad`, which kills changes to the gradient change, since we multiply by nearly zero. This prevents the activation step from resulting in dead neurons. 

>[!important] Approx. gaussian activation layers
>Ideally the activations should be roughly unit gaussian throughout the neural net. 
>One way to achieve this is to scale the input weights. But this strategy starts to fall apart as the NN grows deeper. So, use a normalization layer.
### Activations → normalization layers
**Use a normalization layer** : Batch normalization group normalization / layer normalization. 

More important than proper initialization is batch normalization (2015).  Pre-activation inputs to $\tanh()$ need to be roughly a normalized gaussian, 0 mean and std 1. So just normalize them to be gaussian.

We normalize the batches before passing them into the activation function, so they are unit gaussian during initialization. Then, a gain and bias is applied to the gaussian to allow training to shift and squeeze and otherwise alter the input pre-activation gaussian.

The "problem" with batch normalization is that the examples in the batches are coupled to one another; they are no longer computed independently. This creates variability in the preactivation value and in $h$. This variability is actually helpful for training because it results in an added stochastic effect which reduces overfitting.

Additionally, the bias `b1` does nothing in batch normalization, so may as well remove it. The batch normalization bias `bnbias` takes care of the bias addition.

During normalization, it can also be a good idea to add a small number $\epsilon$ in the denominator to ensure values are well behaved and there is no division by zero.

**NB**: Batch Normalization can be highly error prone due to coupling of examples in forward passes. So, in practice, other kinds of normalization layers like group normalization or layer normalization are used more commonly.
