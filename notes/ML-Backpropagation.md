---
share: "true"
---
Backpropagation works by applying the chain rule iteratively. In an expression like $$\frac{dL}{db} = \frac{de}{db} \cdot \frac{dd}{de} \cdot \frac{dL}{dd}$$we are essentially asking, for a small change in $b$, how does $L$ change? If $b$ is further upstream, more changes in the other nodes will have a downstream effect. Computing the analytical derivative may not be feasible, so we consider a numerical derivative, $$\frac{dL}{db}=\lim_{h \to \infty}\frac{L(b+h)-L(b)}{h}$$
…but often an analytical expression is very easy to implement.

>[!important] It’s just a recursive application of the chain rule


