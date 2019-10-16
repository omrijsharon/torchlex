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
- [x] z.wave(dim)       (returns a normalized ComplexTensor which can be used as a wave function (more information below))
- [x] z.size()          (tensor size)
- [x] len(z)            (tensor length)
- [x] z.euler()         (returns 2 tensors: R and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta" title="\Large \theta" /> in Euler's representation)
- [x] abs(z)            (<img src="https://latex.codecogs.com/svg.latex?\Large&space;z.real^{2}+z.imag^{2}" title="\Large z.real^{2}+z.imag^{2}" />)
- [x] z.magnitude()     (<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sqrt{z.real^{2}+z.imag^{2}}" title="\Large \sqrt{z.real^{2}+z.imag^{2}}" />)
- [x] z.angle()         (Angle of a complex element <img src="https://latex.codecogs.com/svg.latex?\Large&space;(0<\theta<2\pi)" title="\Large (0<\theta<2\pi)" />)
- [x] z.phase()         (Phase of a complex element (can be negative or <img src="https://latex.codecogs.com/svg.latex?\Large&space;>2\pi" title="\Large >2\pi" />))
- [x] z.tensor() or z.z (Get raw torch Tensor)
- [x] z.conj()          (Conjugate)
- [x] z.T or z.t()      (Transpose)
- [x] z.H or z.h()      (Hermitian Conjugate)
- [x] z.requires_grad_()  (same as pytorch's requires_grad_())
### Examples
- [x] Defaults
- [x] 5 ways to create a ComplexTensor
- [x] Using torchlex functions
- [x] Euler representation

Quantum Learning:
- [x] Probability density function
- [x] Wave function

## Additional information
### Probability density function
```
z.PDF(dim)
```
_dim_ plays the same roll as in torch.softmax function.
This function returns the probability density function of your ComplexTensor which is the equivalent of the expectation value in quantum mechanics.
The function divides (normalizes) the ComplexTensor by the sum of abs(z) in  dimension _dim_ and takes the abs of the result.
If left empty or dim=None, the ComplexTensor will be divided by the sum of abs(z) in all dimensions.

### Wave function
```
z.wave(dim)
```
_dim_ plays the same roll as in torch.softmax function.
This function returns a normalized ComplexTensor which is the equivalent of a quantum wave function.
The function divides the ComplexTensor by the sum of abs(z) in  dimension _dim_.
If left empty or dim=None, the ComplexTensor will be divided by the sum of abs(z) in all dimensions.


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

## Examples
In the begining of the code you must import the library:
```
import torchlex
```
### Defaults:
- ComplexTensor default is complex=True. See explanation below.
- ComplexTensor default is requires_grad=True.
### 5 ways to create a ComplexTensor
1. Inserting a tuple of torch tensors or numpy arrays with the same size and dimensions. The first tensor/array will be the real part of the new ComplexTensor and the second tensor/array will be the imaginary part.
```
a = torch.randn(3,5)
b = torch.randn(3,5)
z = torchlex.ComplexTensor((a,b))
```
2. Converting a complex numpy array to a ComplexTensor:
```
z_array = np.random.randn(3,5) + 1j*np.random.randn(3,5)
z = torchlex.ComplexTensor(z_array)
```
3. Inserting a ComplexTensor into ComplexTensor. Completely redundant operation. A waste of computer power. Comes with a warning.
```
z_array = np.random.randn(3,5) + 1j*np.random.randn(3,5)
z_complex = torchlex.ComplexTensor(z_array)
z = torchlex.ComplexTensor(z)
```
4. a. Inserting a torch tensor / numpy array which contains only the real part of the ComplexTensor:
```
x = np.random.randn(3,5)
#or
x = torch.randn(3,5)
z = torchlex.ComplexTensor(x, complex=False)
```
4. b. Inserting a torch tensor which contains the real and the imaginary parts of the ComplexTensor. Last dimension size must be 2.
**Does not work with numpy arrays.**
```
x = np.random.randn(3,5,2)
z = torchlex.ComplexTensor(x, complex=True)
```
5. Inserting a list of complex numbers to ComplexTensor:
```
x = [1, 1j, -1-1j]
z = torchlex.ComplexTensor(x)
```

### Using torchlex functions
exp(log(z)) should be equal to z:
```
x = [1,1j,-1-1j]
z = torchlex.ComplexTensor(x, requires_grad=False)
log_z = torchlex.log(z)
exp_log_z = torchlex.exp(log_z)
```
we get:
```
ComplexTensor([ 1.000000e+00+0.j       , -4.371139e-08+1.j       ,
       -9.999998e-01-1.0000001j], dtype=complex64)
```
which is the original [1,1j,-1-1j] with a small numerical error.

### euler representation
We can get the r and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta" title="\Large \theta" /> of Euler's representation. Lets compare ComplexTensor with Numpy: 
```
x = [1,1j,-1-1j]
z = torchlex.ComplexTensor(x, requires_grad=False)
r, theta = z.euler()
print("ComplexTensor\nr = ", r, '\ntheta = ', theta)
z_np = np.array(x)
print("\nNumpy\nr = ", abs(z_np), '\ntheta = ', np.angle(z_np))
```
we get:
```
ComplexTensor
r =  tensor([1.0000, 1.0000, 1.4142]) 
theta =  tensor([0.0000, 1.5708, 3.9270])

Numpy
r =  [1.         1.         1.41421356] 
theta =  [ 0.          1.57079633 -2.35619449]
```
the last element of theta seems to be different, yet the difference between the two outputs is <img src="https://latex.codecogs.com/svg.latex?\Large&space;2\pi" title="\Large 2\pi" />, which means it is the same angle.
## Quantum Learning
### Probability density function
If z is 2x2 ComplexTensor, then
```
abs_psi = z.PDF()
```
returns a probabilities/Categorical tensor of measuring the ij's state <img src="https://latex.codecogs.com/svg.latex?\Large&space;(|\psi_{ij}|)" title="\Large (|\psi_{ij}|)" />. This Categorical can be samples at will by:
```
abs_psi.sample()
```
### Wave function
If z is 100x5 ComplexTensor, then
```
psi = z.PDF(dim=0)
```
is a collection of 5 wave functions with 100 states each.
This ComplexTensor can be used with Quantum Operators:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\psi^{+}P\hat\psi" title="\Large \psi^{+}P\psi" />
where P is an operator at your choice. For instance, In 1D, <img src="https://latex.codecogs.com/svg.latex?\Large&space;\psi" title="\Large \psi" /> will be a (1D) vector and P will be a (2D) matrix.
