---
share: true
---
https://docs.pytorch.org/docs/2.15/notes/broadcasting.html
``tensor.view`` is an efficient way to adjust how the dimensions are split, without creating a new tensor. 

`BatchNorm1d` : For applying batch normalization over a 2D or 3D input. 
	the kwarg momentum may need to be lowered if the batch size is fairly small
	`affine=True` ensures the batch norm gain and bias are used
	$$y = \frac{x-E[x]}{\sqrt{\text{var}(x)+\epsilon}}*\gamma+\beta$$see [docs : torch.nn.BatchNorm1d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html)