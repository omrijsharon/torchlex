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
- [x] abs(z)
- [x] z.magnitude()
- [x] Angle of a complex element <img src="https://latex.codecogs.com/svg.latex?\Large&space;(0<\theta<2\pi)" title="\Large (0<\theta<2\pi)" />: z.angle()
- [x] Phase of a complex element (can be negative or <img src="https://latex.codecogs.com/svg.latex?\Large&space;<2\pi" title="\Large <2\pi" />): z.phase()
- [x] Get raw torch Tensor: z.tensor() or z.z
- [x] Conjugate: z.conj()
- [x] Transpose: z.T or z.t()
- [x] Hermitian Conjugate: z.H or z.h()
- [x] requires_grad_()
