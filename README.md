# Learning the MMSE Channel Estimator

This code was used to generate the figures in the article

D. Neumann, T. Wiese, and W. Utschick, _Learning the MMSE Channel Estimator_, Accepted for publication in _IEEE Transactions on Signal Processing_, 2018.

## Abstract
We present a method for estimating Gaussian random vectors with random covariance matrices, which uses techniques from the field of machine learning.
Such models are typical in communication systems, where the covariance matrix of the channel vector depends on random parameters, e.g., angles of propagation paths.
If the covariance matrices exhibit certain Toeplitz and shift-invariance structures, the complexity of the MMSE channel estimator can be reduced to _O(M_ log _M)_ floating point operations, where _M_ is the channel dimension.
While in the absence of structure the complexity is much higher, we obtain a similarly efficient (but suboptimal) estimator by using the MMSE estimator of the structured model as a blueprint for the architecture of a neural network.
This network learns the MMSE estimator for the unstructured model, but only within the given class of estimators that contains the MMSE estimator for the structured model.
Numerical simulations with typical spatial channel models demonstrate the generalization properties of the chosen class of estimators to realistic channel models.


## Installation Notes
This code is written in _Julia_ (http://julialang.org) and uses _Julia_'s _TensorFlow_ package, which automatically installs _TensorFlow_ (http://tensorflow.org).
The code was tested with _Julia_ version 0.6, _TensorFlow_ version 1.4.1, and version 0.8 of the _Julia_ package _TensorFlow_.

## License
This code is licensed under 3-clause BSD License:

>Copyright (c) 2018 D. Neumann and T. Wiese
>
>Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
>
>1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
>
>2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
>3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
