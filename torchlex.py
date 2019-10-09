import torch
from torch.distributions import Categorical
import numpy as np
import warnings


def log(z):
    if "ComplexTensor" in z.__class__.__name__:
        r, theta = z.euler()
        result = ComplexTensor((torch.log(r), theta), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.log(z)
    return result


def exp(z):
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        real = torch.exp(a) * torch.cos(b)
        imag = torch.exp(a) * torch.sin(b)
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.exp(z)
    return result


def sin(z):
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        real = torch.sin(a) * torch.cosh(b)
        imag = torch.cos(a) * torch.sinh(b)
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.sin(z)
    return result


def cos(z):
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        real = torch.cos(a) * torch.cosh(b)
        imag = torch.sin(a) * torch.sinh(b)
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.cos(z)
    return result


def tan(z):
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        denominator = torch.cos(2*a) + torch.cosh(2*b)
        real = torch.sin(2*a) / denominator
        imag = torch.sinh(2*b) / denominator
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.tan(z)
    return result


def tanh(z):
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        denominator = torch.cosh(2*a) + torch.cos(2*b)
        real = torch.sinh(2 * a) / denominator
        imag = torch.sin(2 * a) / denominator
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.tanh(z)
    return result


def sigmoid(z):
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        denominator = 1 + 2 * torch.exp(-a) * torch.cos(b) + torch.exp(-2 * a)
        real = 1 + torch.exp(-a) * torch.cos(b) / denominator
        imag = torch.exp(-a) * torch.sin(b) / denominator
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.sigmoid(z)
    return result


def softmax(z, dim):
    '''
    Complex-valued Neural Networks with Non-parametric Activation Functions
    (Eq. 36)
    https://arxiv.org/pdf/1802.08026.pdf
    '''
    if "ComplexTensor" in z.__class__.__name__:
        result = torch.softmax(abs(z), dim=dim)
    else:
        result = torch.softmax(z, dim=dim)
    return result


def CReLU(z):
    '''
    Eq.(4)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        real = torch.relu(a)
        imag = torch.relu(b)
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.relu(z)
    return result

def zReLU(z):
    '''
    Guberman ReLU:
    Nitzan Guberman. On complex valued convolutional neural networks. arXiv preprint arXiv:1602.09046, 2016
    Eq.(5)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        mask = ((0 < z.angle()) * (z.angle() < np.pi/2)).float()
        real = a * mask
        imag = b * mask
        result = ComplexTensor((real,imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.relu(z)
    return result

def modReLU(z, bias):
    '''
    Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.
    Notice that |z| (z.magnitude) is always positive, so if b > 0  then |z| + b > = 0 always.
    In order to have any non-linearity effect, b must be smaller than 0 (b<0).
    '''
    if "ComplexTensor" in z.__class__.__name__:
        a, b = z.real, z.imag
        z_mag = z.magnitude()
        mask = ((z_mag + bias) >= 0).float() * (1 + bias / z_mag)
        real = mask * a
        imag = mask * b
        result = ComplexTensor((real, imag), complex=True, requires_grad=z.requires_grad)
    else:
        result = torch.relu(z)
    return result


class ComplexTensor:
    def __init__(self, x, complex=True, requires_grad=True):
        self.requires_grad = requires_grad
        if 'tuple' in x.__class__.__name__:
            if len(x) == 2:
                if 'ndarray' in x[0].__class__.__name__:
                    ComplexTensor(x[0] + 1j*x[1], requires_grad=self.requires_grad)
                elif 'Tensor' in x[0].__class__.__name__:
                    a = x[0]
                    b = x[1]
                else:
                    raise TypeError("Only torch.tensor or np.array can be converted to a ComplexTensor.")
        elif 'ndarray' in x.__class__.__name__:
            a = torch.from_numpy(x.real).float()
            b = torch.from_numpy(x.imag).float()
        elif 'ComplexTensor' in x.__class__.__name__:
            warnings.warn("Warning: You are trying to convert an already ComplexTensor to a ComplexTensor.")
            self.z = x.z
        elif 'Tensor' in x.__class__.__name__:
            if complex is True:
                if x.size()[-1] == 2:
                    self.z = x
                else:
                    raise RuntimeError('For a Tensor to become complex, the last dimension should be of size 2 and not ' + str(x.size()[-1]) + ". Use: ComplexTensor(x, complex=False).")
            else:
                a = x
                b = torch.zeros_like(x)
        elif 'list' in x.__class__.__name__:
            self.z = ComplexTensor(np.array(x), requires_grad=self.requires_grad).z
        else:
            raise TypeError(x.__class__.__name__ + " cannot be converted to a ComplexTensor.")
        if 'z' not in self.__dict__:
            dim = a.dim()
            self.z = torch.cat((a.unsqueeze(dim),b.unsqueeze(dim)), dim=dim)
        if self.requires_grad:
            self.z = self.z.requires_grad_()

    def requires_grad_(self):
        self.z = self.z.requires_grad_()
        self.requires_grad = True

    def requires_grad_check(self, other):
        return other.requires_grad or self.requires_grad

    @property
    def real(self):
        idx = [slice(None)] * (self.z.dim()-1) + [slice(0, 1)]
        return self.z[idx].squeeze(self.z.dim()-1)

    @property
    def imag(self):
        idx = [slice(None)] * (self.z.dim()-1) + [slice(1, 2)]
        return self.z[idx].squeeze(self.z.dim()-1)

    def __repr__(self):
        # return 'ComplexTensor real part:\n' + "      "+ str(self.real)[6:] + ' \nComplexTensor imaginary part:\n' + "      " + str(self.imag)[6:]
        real = self.real.flatten()
        imag = self.imag.flatten()
        strings = np.asarray([complex(a, b) for a, b in zip(real, imag)]).astype(np.complex64).reshape(*self.size())
        strings = strings.__repr__().replace("array", "ComplexTensor")
        return strings

    def __len__(self):
        return len(self.z)

    def size(self):
        return self.z.size()[:-1]

    def euler(self):
        a, b = self.real, self.imag
        r = torch.sqrt(a**2 + b**2)
        theta = torch.atan(b/a)
        theta[a < 0] += np.pi
        return r, theta

    def __abs__(self):
        return self.real**2 + self.imag**2

    def magnitude(self):
        return torch.sqrt(self.real**2 + self.imag**2)

    def angle(self):
        a, b = self.real, self.imag
        theta = torch.atan(b / a)
        theta[a < 0] += np.pi
        theta = torch.fmod(theta, 2*np.pi)
        return theta

    def phase(self):
        a, b = self.real, self.imag
        theta = torch.atan(b / a)
        theta[a < 0] += np.pi
        return theta

    def tensor(self):
        return self.z

    def __add__(self, other):
        if "ComplexTensor" in other.__class__.__name__:
            result = self.z + other.z
        elif "Tensor" in other.__class__.__name__:
            result = self.z + ComplexTensor(other, requires_grad=self.requires_grad_check(other)).z
        else:
            raise TypeError("ComplexTensor and " + str(other.__class__.__name__) + " cannot be added.")
        return result

    def __radd__(self, other):
        if "Tensor" in other.__class__.__name__:
            result = self.z + ComplexTensor(other, requires_grad=self.requires_grad_check(other)).z
        else:
            raise TypeError(str(other.__class__.__name__) + "and ComplexTensor cannot be added.")
        return result

    def __sub__(self, other):
        if "ComplexTensor" in other.__class__.__name__:
            result = self.z - other.z
        elif "Tensor" in other.__class__.__name__:
            result = self.z - ComplexTensor(other, requires_grad=self.requires_grad_check(other)).z
        else:
            raise TypeError("Cannot subtract " + str(other.__class__.__name__) + " from a ComplexTensor.")
        return result

    def __rsub__(self, other):
        if "ComplexTensor" in other.__class__.__name__:
            result = other.z - self.z
        elif "Tensor" in other.__class__.__name__:
            result = ComplexTensor(other, requires_grad=self.requires_grad_check(other)).z - self.z
        else:
            raise TypeError("Cannot subtract a ComplexTensor from " + str(other.__class__.__name__) + ".")
        return result

    def __truediv__(self, other):
        if "ComplexTensor" in other.__class__.__name__:
            a = self.real
            b = self.imag
            c = other.real
            d = other.imag
            denominator = abs(other)
            real = (a * c + b * d) / denominator
            imag = (b * c - a * d) / denominator
            result = ComplexTensor((real, imag), complex=True, requires_grad=self.requires_grad_check(other))
        elif "Tensor" in other.__class__.__name__:
            result = self / ComplexTensor(other, requires_grad=self.requires_grad_check(other))
        else:
            raise TypeError("ComplexTensor cannot divide " + str(other.__class__.__name__) + ".")
        return result

    def __rtruediv__(self, other):
        if "Tensor" in other.__class__.__name__:
            result = ComplexTensor(other, requires_grad=self.requires_grad_check(other)) / self
        else:
            raise TypeError(str(other.__class__.__name__) + " cannot divide a ComplexTensor.")
        return result

    def __mul__(self, other):
        if "ComplexTensor" in other.__class__.__name__:
            a = self.real
            b = self.imag
            c = other.real
            d = other.imag
            real = a * c - b * d
            imag = a * d + b * c
            result = ComplexTensor((real, imag), requires_grad=self.requires_grad_check(other))
        elif "Tensor" in other.__class__.__name__:
            other = ComplexTensor((other, torch.zeros_like(other)), requires_grad=self.requires_grad_check(other))
            result = self.__matmul__(other)
        else:
            raise TypeError("ComplexTensor cannot multiply " + str(other.__class__.__name__))
        return result

    def __matmul__(self, other):
        if "ComplexTensor" in other.__class__.__name__:
            a = self.real
            b = self.imag
            c = other.real
            d = other.imag
            real = a @ c - b @ d
            imag = a @ d + b @ c
            result = ComplexTensor((real, imag), requires_grad=self.requires_grad_check(other))
        elif "Tensor" in other.__class__.__name__:
            other = ComplexTensor((other, torch.zeros_like(other)), requires_grad=self.requires_grad_check(other))
            result = self.__matmul__(other)
        else:
            raise TypeError("ComplexTensor cannot matrix - multiply " + str(other.__class__.__name__))
        return result

    def __rmul__(self, other):
        if "Tensor" in other.__class__.__name__:
            other = ComplexTensor((other, torch.zeros_like(other)), requires_grad=self.requires_grad_check(other))
            result = self.__mul__(other)
        else:
            raise TypeError('Cannot multiply ' + str(other.__class__.__name__) + " with a ComplexTensor.")
        return result

    def __rmatmul__(self, other):
        if "Tensor" in other.__class__.__name__:
            other = ComplexTensor((other, torch.zeros_like(other)), requires_grad=self.requires_grad_check(other))
            result = self.__matmul__(other)
        else:
            raise TypeError('Cannot multiply ' + str(other.__class__.__name__) + " with a ComplexTensor.")
        return result

    def conj(self): #conjugate
        a, b = self.real, self.imag
        return ComplexTensor((a, -b), complex=True)

    def t(self):
        return self.T

    def h(self):
        return self.H

    def PDF(self, dim=None): #Probability density function
        z_abs = self.__abs__()
        if dim is None:
            result = z_abs/torch.sum(z_abs)
        else:
            result = z_abs / torch.sum(z_abs, dim=dim)
        return Categorical(result)

    def wave(self, dim=None):
        z_abs = self.__abs__()
        if dim is None:
            result = self.z / torch.sum(z_abs)
        else:
            result = self.z / torch.sum(z_abs, dim=dim)
        return ComplexTensor(result, True)

    @property
    def T(self): #transpose
        a, b = self.real, self.imag
        return ComplexTensor((a.t(), b.t()), True)

    @property
    def H(self): #hermitian conjugate
        a, b = self.real, self.imag
        return ComplexTensor((a.t(), -b.t()), True)