---
share: true
---
How to find network parameters that minimize the loss. After choosing [ML-parameter-initialization](ML-parameter-initialization.md), the gradients with respect to the parameters are computed and the parameters are adjusted using the gradients such that the loss is decreased. 
### Gradient descent
Optimization algorithms find parameters to minimize the loss,$$\hat{\phi}=\text{argmin}_\phi \left[ L[\phi]\right]$$Assume we work with the input parameters $\phi =[\phi_0, \phi_1, ..., \phi_N]^T$. In gradient descent, we follow 2 steps.
1. compute the derivatives of the loss wrt the params,$$\frac{\partial L}{\partial \phi} = \begin{bmatrix} \frac{\partial L}{\partial \phi_1} \\ \frac{\partial L}{\partial \phi_2} \\ \vdots \\ \frac{\partial L}{\partial \phi_N} \\ \end{bmatrix}$$
2. update the params, using the learning rate $\alpha$ (positive scalar),$$\phi^{\text{update}}=\phi-\alpha\cdot \frac{\partial L}{\partial \phi}$$
Recall that the gradient points in the direction of the maximum slope, and the negative points in the minimization direction. The hyperparameter $\alpha$ can be optimized with a *line search procedure*. Training can be stopped once the gradient is sufficiently small.

**Linear regression**: Gradient descent can be applied to solve the parameters of a linear regression problem, where the least squares function is used as the [ML-Loss-Functions](ML-Loss-Functions.md). Note that linear regressions are “convex”, meaning they always have a single global minimum; we cannot get trapped in a local minimum. 
**Gabor model**: Neural nets are not generally convex, so we need to consider a simple nonlinear model. The Gabor model is a 2 parameter, non-linear model. Again, the least square function can be used for the loss.
**Local minima & saddle points**: The Gabor model has multiple local minima, so it is possible to apply gradient descent without reaching the global minimum from a random initialization. Additionally, since the gradient is small around saddle points, there is a chance of prematurely ending the optimization. 
**Newton method** takes curvature into account to slow down steps at steep points as an improvement on gradient descent. 

>[!warning] Optimization on gradient descent…
>can depend heavily on the starting point. 

- - -
### Stochastic gradient descent

Its challenging to find the global minimum of a high-dimensional, non-linear function. Stochastic gradient descent (SGD) addresses the initialization dependence inherent in gradient descent to some extent by ==🟢adding noise to the gradient at each step==. This offers the *chance* to exit a shallow valley. 

To introduce randomness, the gradient of a small **batch** of the training data is used for the gradient. Batches are typically pulled without replacement until all of the training data is used, and then resamples through the training set again. Each pass though the training data is called an **epoch**. Alternatively, a different loss function may be applied at each iteration. 

With SGD, the updates are essentially sensible and non-optimal. It is also much cheaper to optimize over batches than over the full training set and it is *possible* to escape local minima. It is probably more helpful for avoiding a trap at a saddle point. SGD is also somewhat less susceptible to overfitting than gradient descent. 

In practice, using a *learning rate schedule* is common while training with SGD. The learning rate is gradually decreased every $N$ epochs. 

NB: there is a relationship between the ratio of the batch size to learning rate. 
### Momentum
Momentum is a common modification in which the updates are weighted by the past gradients, not just the current. The momentum $\mathbf{m}$ is computed using the previous momentum and a decaying sum of previous gradients,$$\phi^{\text{update}}=\phi-\alpha\cdot \mathbf{m}$$the effect is that a consistent gradient points in one way and accumulates over steps, while flipping signs tend to cancel out. This allows for faster convergence along a level floor: a smooth trajectory. The momentum update contains the decay within the simple update,$$m_t = \beta\cdot m_{t-1}+(1-\beta)\cdot g_t$$this works because of recursion.

Nesterov accelerated momentum is a better estimation, since it computes the gradient at the next step, rather than the current point. This is better for correcting overshooting and better theoretical backing for convergence. 

All prior gradients contribute to the current update, but the influence of shrinks geometrically with age. 
### Adam : adaptive moment estimation
<u>The problem</u> : It can be tricky to tune the learning rate, especially because the gradient maybe be steep in some directions and shallow in others. Where the gradient is steep, it may be wise to explore further which is prevented by the small gradient. Where the gradient is high, smaller steps are more cautious and avoid overstepping rough/steep terrain. 
<u>Approach</u> : To make progress in multiple directions, regardless of the gradient magnitude, normalize the gradient and move a fixed distance in each direction. The learning rate governs the fixed distance. Two averages for each parameter are tracked: the momentum and the second momentum. Adam combines momentum smoothing and parameter scaling.
- ==first momentum== is essentially the mean of the gradients in time, where the more recent gradients are weighted more strongly. $$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$$
- ==second momentum== is for normalizing the magnitude, $$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2$$
- ==🔵bias correction== rescales these averages because $m$ and $v$ near $t=0$ is zero, $$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$this bias goes to zero after sufficient steps contribute to the average. 
- ==🟢adam update== is normalized in magnitude and steps in the direction opposite to th magnitude,$$\theta_t = \theta_{t-1}-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}}+\varepsilon}$$ where $\varepsilon$ is a small constant to prevent zero division. 
This algorithm converges and makes progress in all the parameter directions. In practice, is normally used in a stochastic setting, due to mini-batching. Adam also helps to correct for parameter gradient’s dependence on NN depth. Complex learning rate schedules are normally less relevant. 
The most popular adam algorithm: Adam (Kingma & Ba, 2015)

>[!abstract] References
>See Chapter 6 in [@princeUnderstandingDeepLearning](@princeUnderstandingDeepLearning.md)
>	*Reviews of optimization algorithms for neural networks can be found in Ruder (2016), Bottou et al. (2018), and Sun (2020).*

**The Hessian** is a matrix of the second derivatives. The eigenvalues of the Hessian at points where the gradient is zero allows classification of the point as, 
1) minimum, if all eigenvalues are positive
2) maximum, if all eigenvalues are negative
3) saddlepoint, if there is a mixture of positive and negative eigenvalues
