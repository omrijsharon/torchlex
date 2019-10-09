# torchlex
Complex tensor and complex functions for pytorch.

Inspired by https://github.com/williamFalcon/pytorch-complex-tensor.

Based on:
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
- [ ] sigmoid(z)
- [ ] tanh(z)
- [ ] softmax(z)
### ReLU function versions for complex numbers
- [x] CReLU(z)
- [x] zReLU(z)
- [x] modReLU(z, bias)
### ComplexTensor Operation
- [x] addition (z + other and other + z)
- [x] division (z / other and other / z)
- [x] multiplication (z * other and other * z)
- [x] matrix multiplication (z @ other and other @ z)
### ComplexTensor Functions and Properties
- [x] z.real
- [x] z.imag
- [x] z.size()
- [x] z.euler()
- [x] len(z)
- [x] abs(z)            (<img src="https://latex.codecogs.com/svg.latex?\Large&space;z.real^{2}+z.imag^{2}" title="\Large z.real^{2}+z.imag^{2}" />)
- [x] z.magnitude()     (<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sqrt(z.real^{2}+z.imag^{2})" title="\Large \sqrt(z.real^{2}+z.imag^{2})" />)
- [x] z.angle()         (Angle of a complex element <img src="https://latex.codecogs.com/svg.latex?\Large&space;(0<\theta<2\pi)" title="\Large (0<\theta<2\pi)" />)
- [x] z.phase()         (Phase of a complex element (can be negative or <img src="https://latex.codecogs.com/svg.latex?\Large&space;<2\pi" title="\Large <2\pi" />))
- [x] z.tensor() or z.z (Get raw torch Tensor)
- [x] z.conj()          (Conjugate)
- [x] z.T or z.t()      (Transpose)
- [x] z.H or z.h()      (Hermitian Conjugate)
- [x] requires_grad_()
