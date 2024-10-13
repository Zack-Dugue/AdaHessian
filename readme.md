# AdaHessian

This is an unofficial pytorch implementation of the [AdaHessian Optimizer](https://arxiv.org/abs/2006.00719). It is largely based on @davada54's [implementation](https://github.com/davda54/ada-hessian/tree/master). With a few interesting changes with motivated the creation of this repo. 

## Improvements from Base AdaHessian
There are 2  significant improvements from Base Ada Hessian in this repo. 

### We perform momentum over the Hessian, rather than the squared Hessian.
In the original Ada Hessian Paper the paper writers chose to have the Ada Hessian Algorithm track the exponential moving average of the *square* of the Hessian. That value is then plugged in to replace what is generally the second moment in the Adam Optimizer. In this repo, I track the Hessian Directly. <br/>

_Original Paper Formula_:<br/>
$$v_{t+1} = v_t \cdot (1 - \beta_1 ) + H_{diag}^2 \cdot \beta_1$$ <br/>
_Our formula_:<br/>
$$v_{t+1} = v_{t} \cdot (1 - \beta_1 ) +  H_{diag}^2 \cdot  \beta_1$$ <br/>
Obviously this $v_t$ now can potentially contain negative values, and thus we take the absolute value before plugging it into the Adam formula. I found that accumulating the Hessian directly had better results than the squared method from the paper. I believe this is because this $v_t$ is closer to the true hessian and is also larger than the value tracked in the paper. 

### We use a Control Variate Method for the Hutchinson Estimator:

Control Variate Methods are a way of reducing the variance of a monte carlo estimate. They are computed by designating a "control variate" random variable whose mean is known and that is correlated with the random variable whose expectation we're trying to evaluate. They're often used in the trace estimation component of Hutchinson's Algorithm, a good resource explaining this can be found [here](https://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html) . For trace estimation algorithms, they estimate the diagonal of the hessian using the prior monte carlo samples, and use that as the control variate. In this implementation, we have the control variate be our momentum estimate of the Hessian itself. In theory, this shouldn't work, since that variable isn't actually random, it's a known value. In practice though, doing this actually yields superior generalization results. <br/>
To compute this, instead of making our output values in the autograd.grad computation the gradients, we make the output values the gradients minus the control variate times the weights. At some point I might refine this readme further to show these derivation, but the final result there is something matching the Contorl Variate formulation for the Hutchinsons Estimator.
