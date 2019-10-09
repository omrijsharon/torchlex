# torchlex
Pytorch extension for Complex tensors and complex functions.

Inspired by https://github.com/williamFalcon/pytorch-complex-tensor.

Based on the papers:
- Deep Complex Networks (https://arxiv.org/pdf/1705.09792.pdf)
- Unitary Evolution Recurrent Neural Networks (https://arxiv.org/pdf/1511.06464.pdf)
- On Complex Valued Convolutional Neural Networks (https://arxiv.org/pdf/1602.09046.pdf)

## Table of Content:
### Functions
- [x] exp(z)
- [x] log(z)
- [x] sin(z)
- [x] cos(z)
- [x] tan(z)
- [x] tanh(z)
- [x] sigmoid(z)
- [x] softmax(z)
### ReLU function versions for complex numbers
More information in the documentation below
- [x] CReLU(z)
- [x] zReLU(z)
- [x] modReLU(z, bias)
### ComplexTensor Operation
- [x] addition (z + other and other + z)
- [x] subtraction (z - other and other - z)
- [x] multiplication (z * other and other * z)
- [x] matrix multiplication (z @ other and other @ z)
- [x] division (z / other and other / z)
### ComplexTensor Functions and Properties
- [x] z.real            (real part of z)
- [x] z.imag            (imaginary part of z)
- [x] z.PDF(dim)        (Probability density function, more information in the documentation below)
- [x] z.wave(dim)       (returning a normalized ComplexTensor which can be used as a wave function (more information below))
- [x] z.size()          (tensor size)
- [x] len(z)            (tensor length)
- [x] z.euler()         (returns 2 tensors: R and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta" title="\Large \theta" /> in Euler's representation)
- [x] abs(z)            (<img src="https://latex.codecogs.com/svg.latex?\Large&space;z.real^{2}+z.imag^{2}" title="\Large z.real^{2}+z.imag^{2}" />)
- [x] z.magnitude()     (<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sqrt{z.real^{2}+z.imag^{2}}" title="\Large \sqrt{z.real^{2}+z.imag^{2}}" />)
- [x] z.angle()         (Angle of a complex element <img src="https://latex.codecogs.com/svg.latex?\Large&space;(0<\theta<2\pi)" title="\Large (0<\theta<2\pi)" />)
- [x] z.phase()         (Phase of a complex element (can be negative or <img src="https://latex.codecogs.com/svg.latex?\Large&space;<2\pi" title="\Large <2\pi" />))
- [x] z.tensor() or z.z (Get raw torch Tensor)
- [x] z.conj()          (Conjugate)
- [x] z.T or z.t()      (Transpose)
- [x] z.H or z.h()      (Hermitian Conjugate)
- [x] z.requires_grad_()  (same as pytorch's requires_grad_())

## Additional information
### Probability density function
```
z.PDF(dim)
```
_dim_ plays the same roll as in torch.softmax function.
This function returns the probability density function of your ComplexTensor which is the equivalent of the expectation value in quantum mechanics.
The function divides (normalizes) the ComplexTensor by the sum of abs(z) in  dimension _dim_ and takes the abs of the result.
If left empty or dim=None, the ComplexTensor will be divided by the sum of abs(z) in all dimentions.

### Wave function
```
z.wave(dim)
```
_dim_ plays the same roll as in torch.softmax function.
This function returns a normalized ComplexTensor which is the equivalent of a quantum wave function.
The function divides the ComplexTensor by the sum of abs(z) in  dimension _dim_.
If left empty or dim=None, the ComplexTensor will be divided by the sum of abs(z) in all dimentions.


### Softmax
Eq.(36) in the paper Complex-valued Neural Networks with Non-parametric Activation Functions

https://arxiv.org/pdf/1802.08026.pdf

Simone Scardapane, Steven Van Vaerenbergh, Amir Hussain and Aurelio Uncini

### ReLU function versions for complex numbers
#### CReLU(z)
Deep Complex Networks Eq.(5).

https://arxiv.org/pdf/1705.09792.pdf

Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio & Christopher J Pal

#### zReLU(z)
Pages 15-16 in the dissertation: On complex valued convolutional neural networks.

https://arxiv.org/pdf/1602.09046.pdf

Nitzan Guberman, Amnon Shashua.

Also refered as Guberman ReLU in Deep Complex Networks Eq.(5) (https://arxiv.org/pdf/1705.09792.pdf).

    
#### modReLU(z, bias)
Eq.(8) in the paper: Unitary Evolution Recurrent Neural Networks

https://arxiv.org/pdf/1511.06464.pdf
  
Martin Arjovsky, Amar Shah, and Yoshua Bengio.

Notice that |z| (z.magnitude) is always positive, so if b > 0  then |z| + b > = 0 always.
In order to have any non-linearity effect, b must be smaller than 0 (b<0).
