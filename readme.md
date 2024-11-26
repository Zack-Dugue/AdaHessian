# AdaHessian

This is an unofficial pytorch implementation of the [AdaHessian Optimizer](https://arxiv.org/abs/2006.00719). It is largely based on @davada54's [implementation](https://github.com/davda54/ada-hessian/tree/master). With a few interesting changes with motivated the creation of this repo. 

## Improvements from Base AdaHessian
There are 2  significant improvements from Base Ada Hessian in this repo. 

### We compute the (momentum) EMA over the Hessian, rather than the squared Hessian.
AdaHessian can be thought of as Adam, but it replaces the squared gradient in the denominator, with a term computed from the Hessian. In the original Ada Hessian Paper the paper writers chose to have the Ada Hessian Algorithm track the exponential moving average of the *square* of the Hessian. That value is then plugged in to replace what is generally the second moment in the Adam Optimizer. In this repo, I track the EMA of the Hessian directly. <br/>

_Original Paper Formula_:<br/>
$$v_{t+1} = v_t \cdot (1 - \beta_1 ) + H_{diag}^2 \cdot \beta_1$$ <br/>
_Our formula_:<br/>
$$v_{t+1} = v_{t} \cdot (1 - \beta_1 ) +  H_{diag} \cdot  \beta_1$$ <br/>

Obviously this $v_t$ now can potentially contain negative values, and thus we take the absolute value before plugging it into the Adam formula. I found that accumulating the Hessian directly had better results than the squared method from the paper. I believe this is because this $v_t$ is closer to the true hessian and is also larger than the value tracked in the paper. 

### We use a Control Variate Method for the Hutchinson Estimator:

Control Variate Methods are a way of reducing the variance of a monte carlo estimate. They are computed by designating a "control variate" random variable whose mean is known and that is correlated with the random variable whose expectation we're trying to evaluate. They're often used in the trace estimation component of Hutchinson's Algorithm, a good resource explaining this can be found [here](https://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html) . For trace estimation algorithms, they estimate the diagonal of the hessian using the prior monte carlo samples, and use that as the control variate. In this implementation, we have the control variate be our running EMA of the Hessian itself ($v_{t+1}$).
To compute this, instead of making our output values in the autograd.grad computation the gradients, we make the output values the gradients minus the control variate times the weights. 
<br/>
The standard way of computing the Hessian Vector Product in Ada Hessian.:

$$\frac{\partial}{\partial \theta}  \left( \left(\frac{\partial \mathcal{L}}{\partial \theta}\right)^T z\right) = $$

$$\frac{\partial ^2 \mathcal{L}}{\partial \theta} z + \left(\frac{\partial \mathcal{L}}{\partial \theta}\right)^T \left(\frac{\partial}{\partial \theta} z\right)  = $$

$$ Hz + 0 $$

Then averaged over monte carlo samples of $z$ we get:
$$H_{diag} = \mathbb{E}[z \odot  Hz] $$
Where $z$ is sampled according to a radermacher distribution. 
<br/><br/>

To take advantage of the control variate method we modify this equation: Note that $v_t$ as computed above, can be thought of as a diagonal matrix, since it is an exponential moving average of estimates of a diagonal matrix. It is stored as a vector in the code, but for mathematical purposes it is a diagonal matrix. 

$$\frac{\partial}{\partial \theta}  \left( \left(\frac{\partial \mathcal{L}}{\partial \theta} - v_t \theta \beta_1 \right)^T z\right) = $$

$$\left(\frac{\partial ^2 \mathcal{L}}{\partial \theta} - v_t \beta_1\right) z +\left(\frac{\partial \mathcal{L}}{\partial \theta}  - v_t \theta \beta_1\right)^T \left(\frac{\partial}{\partial \theta} z\right)  = $$ 

$$ (H-v_t \beta_1) z + 0 $$ 
Then averaged over monte carlo samples of $z$ we get:
$$H_{diag} = \mathbb{E}[z \odot  (H - v_t \beta_1)z] $$
Where $z$ is sampled according to a radermacher distribution. The above is the equation for our a control variate estimate of the hessian diagonal, where $v_t \beta_1$ is our estimate for the value hessian diagonal. 

## What this repo contains:
The AdaHessian.py file contains the actual optimizer. The example.ipynb is a google collab notebook that allows you to evaluate the optimizer on a vision (cifar10 - resnet 50) and nlp ( wikitext2 - default pytorch transformer). One thing worth noting is that the increased accuracy of the Hessian estimate is only really useful if you reduce the momentum for the hessian diagonal. The default is still .999 (matching with Adam and the original Ada Hessian implementation), but it's something you can mess with yourself. 
