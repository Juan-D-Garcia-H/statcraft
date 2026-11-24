# Third-party notices

The statcraft source code is licensed under the Apache License 2.0 (see
`LICENSE`). Some C++ components are ports or adaptations of third-party
software distributed under BSD-style licenses. The copyright notices, license
terms and citations below are reproduced so that redistributions of statcraft
comply with those licenses.

## L-BFGS-B 3.0

**Files:** `include/minimizer/lbfgsb.hpp` and `include/minimizer/lbfgsb/*.hpp`

This is a C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
Richard Byrd, Jorge Nocedal and Jose Luis Morales.

Upstream: <https://users.iems.northwestern.edu/~nocedal/lbfgsb.html>

Copyright (c) 1990-2007, Ciyou Zhu, Richard Byrd, Jorge Nocedal, Jose Luis
Morales. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Papers to cite:

- R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, "A limited memory algorithm for
  bound constrained optimization", SIAM Journal on Scientific Computing,
  16(5):1190-1208, 1995.
- C. Zhu, R. H. Byrd and J. Nocedal, "L-BFGS-B: Algorithm 778: L-BFGS-B,
  FORTRAN routines for large scale bound constrained optimization", ACM
  Transactions on Mathematical Software, 23(4):550-560, 1997.

## SciPy optimizers and MINPACK-2 line search

**Files:** `include/minimizer/bfgs.hpp`, `include/minimizer/nelder_mead.hpp`,
`include/saemfit/nelder_mead.hpp`, and the `dcsrch`/`dcstep` line-search
routines under `include/minimizer/lbfgsb/`.

These are C++ ports of `scipy.optimize.minimize(method="BFGS")` and
`scipy.optimize.minimize(method="Nelder-Mead")`. The BFGS strong-Wolfe line
search derives from the MINPACK-2 `dcsrch`/`dcstep` algorithms as packaged by
SciPy.

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Papers and references to cite:

- P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
  computing in Python", Nature Methods 17, 261-272, 2020.
- J. A. Nelder and R. A. Mead, "A simplex method for function minimization",
  The Computer Journal 7(4):308-313, 1965.
- J. J. More and D. C. Sorensen, "Computing a trust region step", SIAM Journal
  on Scientific and Statistical Computing 4(3):553-572, 1983.

## Build-time dependencies

CMake uses system copies of the following libraries when available and fetches
their sources otherwise. They are not vendored in the statcraft source tree or
in the wheel, but their code is used when compiling the extension modules:

- **Eigen 3.4.0** - MPL-2.0 - <https://eigen.tuxfamily.org/>
- **Boost.Math 1.84.0** - BSL-1.0 - <https://www.boost.org/doc/libs/1_84_0/libs/math/>
- **nanobind 2.4.0** - BSD-3-Clause - <https://github.com/wjakob/nanobind>
