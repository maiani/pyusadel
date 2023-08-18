# Usadel Equations

## Introduction to Usadel equations
In the time-reversed hole basis, $\psi = (\psi_\uparrow, \psi_\downarrow, -\psi_\downarrow, \psi_\uparrow)$, the Usadel equation in Matsubara representation reads
$$
D \nabla\cdot(\check{g}\nabla\check{g}) - [\omega_n \tau_3 \sigma_0 + i \boldsymbol{h}\cdot\boldsymbol{\sigma} \tau_3 + \check{\Delta} +\check{\Sigma}, \check{g}] = 0
    $$
that needs to be complemented with the normalization constraint $\check{g}^2 = 1$. 
Moreover, since we are considering a conventional singlet s-wave pairing mechanism, we set $\check{\Delta}= \Delta \tau_1$.

In the self-energy term, we include spin- and charge-imbalance relaxation due to spin-flipping scattering with magnetic impurities and spin-orbit $\Sigma = \Sigma_\mathrm{sf} + \Sigma_\mathrm{so}$ with
\begin{align}
\Sigma_\mathrm{sf}&=\frac{\boldsymbol{\sigma}\cdot\tau_3\check{g}\boldsymbol{\sigma} \tau_3}{8 \tau_\mathrm{sf}} \\
\Sigma_\mathrm{so}&=\frac{\boldsymbol{\sigma}\cdot\check{g}\boldsymbol{\sigma}}{8 \tau_\mathrm{so}}
\end{align}

Relaxation due to orbital-depairing can be included adding the self-energy
\begin{align}
\Sigma_\mathrm{ob}&=\frac{\tau_3\check{g}\tau_3}{\tau_\mathrm{ob}} 
\end{align}
with $\tau_{orb}^{-1}= D e^2 B^2 d^2 / 6$

## Ivanov-Fominov parametrization

We adopt the $(\theta, \boldsymbol{M})$ in which the quasiclassical propagator reads
$$
\check{g} = (\cos\theta M_0 \sigma_0 + i \sin\theta \boldsymbol{M}\cdot \boldsymbol{\sigma}) \tau_3 +( \sin\theta M_0 \sigma_0 - i \cos\theta \boldsymbol{M}\cdot\boldsymbol{\sigma})\tau_1
$$
with $M_0^2 - \mathbf{M}^2 = 1$.

In this parametrization, the Usadel equation can be split in a pair of scalar (the $i\tau_2 \sigma_0$ component). and a vector (the $\tau_2 \boldsymbol{\sigma}$) coupled PDEs
$$
D\nabla^2\theta+ 2M_0(\Delta\cos\theta-\omega_n\sin\theta) - 2 \cos\theta \boldsymbol{h}\cdot\boldsymbol{M}-\frac{(2 M_0^2 + 1) \sin(2\theta)}{4\tau_\mathrm{sf}}-2\frac{(2 M_0^2 - 1) \sin(2\theta)}{\tau_\mathrm{ob}}=0,
$$

$$
D\left(\boldsymbol{M}\nabla^2M_0  - M_0\nabla^2\boldsymbol{M}\right) +2 \boldsymbol{M} (\Delta \sin\theta + \omega_n \cos\theta) - 2  \sin\theta \boldsymbol{h} M_0  + \Big[\frac{1}{\tau_\mathrm{so}}+ \big(\frac{1}{2 \tau_\mathrm{sf}}+\frac{4}{\tau_\mathrm{ob}}\big)\cos(2\theta)\Big] M_0 \mathbf{M}= 0.
$$
while the gap equation reads 
$$
\Delta =  \gamma \int_{-\omega_D}^{+\omega_D} \frac{\mathrm{tr}{\tau_x\sigma_0 g}}{2}
$$
for $ \gamma = N_0 V_0$. This can be recasted in the form
$$\Delta \log \left(\frac{T}{T_{c0}}\right) = 2\pi T \sum_{\omega_n>0}M_0 \sin\theta  - \frac{\Delta}{\omega_n}$$

With this parametrization, all the field involved are real in the imaginary axis. The real axis can be obtained with the substitution $\omega_n \to - i \omega$.

To evaluate the stability of the superconductive phase, we need to calculate the free energy that reads
$$
\begin{split}
f_\mathrm{sn} = N_0 \pi T \sum_{\omega_n>0} \mathrm{tr} \Big\{&(\omega_n + i \boldsymbol{h}\cdot\boldsymbol{\sigma})(1-\tau_z g) - \frac{1}{2}(\Delta \tau_+ + \Delta^\dagger \tau_-)g + \frac{D}{4}(\nabla\check{g})^2 \\
&+\frac{1}{16 \tau_\mathrm{so}}\Big[3 - (\boldsymbol{\sigma} \check{g}) \cdot (\boldsymbol{\sigma} \check{g})]
+ \frac{1}{16 \tau_\mathrm{sf}}[3-(\boldsymbol{\sigma}\tau_3\check{g})\cdot(\boldsymbol{\sigma}\tau_3\check{g})] \Big\}
\end{split}
$$

$$
\begin{split}
f_\mathrm{sn} = N_0 \pi T \sum_{\omega_n>0} \Big\{ &4 \omega_n - 2 M_0 (2 \omega_n \cos\theta + \Delta\sin\theta) + 4 (M_x h_x + M_y h_y) \sin\theta + \\
&D \left[(\nabla \theta)^2+ (\nabla M_0)^2 - (\nabla M_x)^2 - (\nabla M_y)^2 \right] + \\
&\frac{1}{4}\Big[3\Big(\tau_\mathrm{so}^{-1} + \tau_\mathrm{sf}^{-1}\Big) - 3\Big(\tau_\mathrm{so}^{-1} + \tau_\mathrm{sf}^{-1}\cos(2\theta)\Big)\Big] M_0^2
-\Big(\tau_\mathrm{so}^{-1} - \tau_\mathrm{sf}^{-1}\Big)\cos(2\theta) (M_x^2+M_y^2+M_z^2)
\Big\}
\end{split}
$$

## Numerical method

Since $\mathbf{h}$ lies in the $(x, y)$ plane, $M_z = 0$. We then use the normalization condition $M_0^2  = \sqrt{ 1 + M_x^2 + M_y^2}$ to get rid of $M_0$ too. Therefore the coupled system of PDEs reads

$$
\begin{cases}
f_0(\theta, M_x, M_y) = 0\\
f_1(\theta, M_x, M_y) = 0\\
f_2(\theta, M_x, M_y) = 0
\end{cases}
$$
where

$$
f_0 = D\nabla^2\theta +2 M_0 ( \Delta\cos\theta - \omega_n \sin\theta) - 2  (h_x M_x + h_y M_y)\cos\theta-\Big[ \frac{(2 M_0^2 + 1)}{4\tau_\mathrm{sf}}+2\frac{(2 M_0^2 - 1)}{\tau_\mathrm{ob}} \Big] \sin(2\theta)=0,
$$

$$
\begin{split}
f_1 = &D\left(M_x \nabla^2 M_0 -  M_0 \nabla^2 M_x \right)
+2 M_x ( \Delta \sin\theta +\omega_n\cos\theta) - 2  h_x M_0\sin\theta + \\
&\Big[\frac{1}{\tau_\mathrm{so}}+ \big(\frac{1}{2 \tau_\mathrm{sf}}+\frac{4}{\tau_\mathrm{ob}}\big)\cos(2\theta)\Big]  M_0 M_x= 0,
\end{split}
$$

$$
\begin{split}
f_2 =  &D\left(M_y \nabla^2 M_0 -  M_0 \nabla^2 M_y \right)
+2 M_y ( \Delta \sin\theta + \omega_n\cos\theta) - 2 h_y M_0 \sin\theta + \\
&\Big[\frac{1}{\tau_\mathrm{so}}+ \big(\frac{1}{2 \tau_\mathrm{sf}}+\frac{4}{\tau_\mathrm{ob}}\big)\cos(2\theta)\Big]  M_0 M_y = 0.
\end{split}
$$


To solve the fully coupled problem, we use Newton's method. We define the next iteration as $\theta_{t+1} = \theta_{t} + \tilde{\theta}$ and similarly for $M_x$ and $M_y$. By setting
$$
\begin{cases}
f_0(\theta_{t} + \tilde{\theta}, M_x + \tilde{M_x}, M_y + \tilde{M_y}) = 0\\
f_1(\theta_{t} + \tilde{\theta}, M_x + \tilde{M_x}, M_y + \tilde{M_y}) = 0\\
f_2(\theta_{t} + \tilde{\theta}, M_x + \tilde{M_x}, M_y + \tilde{M_y}) = 0
\end{cases}
$$
one gets
$$
\begin{pmatrix}
\frac{\delta f_0}{\delta \theta} & \frac{\delta f_0}{\delta M_x} & \frac{\delta f_0}{\delta M_y} \\
\frac{\delta f_1}{\delta \theta} & \frac{\delta f_1}{\delta M_x} & \frac{\delta f_1}{\delta M_y} \\
\frac{\delta f_2}{\delta \theta} & \frac{\delta f_2}{\delta M_x} & \frac{\delta f_2}{\delta M_y} \\
\end{pmatrix}
\begin{pmatrix}
\tilde{\theta} \\
\tilde{M}_x \\
\tilde{M}_y \\
\end{pmatrix} 
= 
\begin{pmatrix}
-f_0(\theta_t, M_{x, t}, M_{y, t} ) \\
-f_1(\theta_t, M_{x, t}, M_{y, t} ) \\
-f_2(\theta_t, M_{x, t}, M_{y, t} ) \\
\end{pmatrix}
$$
where

$$
\frac{\delta f_0}{\delta \theta} = (D\nabla^2+ 2 M_0\left(-\Delta\sin\theta - \omega\cos\theta\right)+ 2 (h_x M_x + h_y M_y) \sin\theta - \Big[ \frac{2M_0^2 + 1}{2 \tau_\mathrm{sf}} + 4\frac{(2 M_0^2 - 1)}{\tau_\mathrm{ob}}\Big]\cos(2\theta)
$$


$$
\frac{\delta f_0}{\delta M_x} = 2 \frac{M_x}{M_0} (\Delta\cos\theta - \omega\sin\theta)   - 2 h_x \cos\theta -  \Big[\frac{1}{\tau_\mathrm{sf}} + \frac{8}{\tau_\mathrm{ob}}\Big] M_x \sin(2\theta)
$$


$$
\frac{\delta f_0}{\delta M_y} = 2 \frac{M_y}{M_0} (\Delta\cos\theta - \omega\sin\theta)   - 2 h_y \cos\theta - \Big[\frac{1}{\tau_\mathrm{sf}} + \frac{8}{\tau_\mathrm{ob}}\Big]  M_y \sin(2\theta)
$$

$$
\frac{\delta f_1}{\delta \theta} = 2 M_x(\Delta\cos\theta - \omega_n\sin\theta) - 2 h_x M_0 \cos \theta - \Big[\frac{1}{\tau_\mathrm{sf}} + \frac{8}{\tau_\mathrm{ob}}\Big] M_0 M_x \sin(2\theta)
$$

$$
\begin{split}
\frac{\delta f_1}{\delta M_x} = &D\left( \nabla^2 M_0 + M_x \nabla^2 \frac{M_x}{M_0} - \frac{M_x}{M_0} \nabla^2 M_x -M_0 \nabla^2  \right) +2( \Delta \sin\theta + \omega_n \cos\theta) \\
&-2 \sin\theta h_x \frac{M_x}{M_0} + \Big[\frac{1}{\tau_\mathrm{so}} + \Big(\frac{1}{2 \tau_\mathrm{sf}}+ \frac{4}{\tau_\mathrm{ob}}\Big)\cos(2 \theta) \Big] \Big(\frac{M_x^2}{M_0} + M_0\Big)
\end{split}
$$

$$
\frac{\delta f_1}{\delta M_y} = D\left( M_x \nabla^2 \frac{M_y}{M_0} -  \frac{M_y}{M_0} \nabla^2 \right) -2  \sin\theta h_x \frac{M_y}{M_0} + \Big[\frac{1}{\tau_\mathrm{so}} + \Big(\frac{1}{2 \tau_\mathrm{sf}} + \frac{4}{\tau_\mathrm{ob}}\Big)\cos(2 \theta)\Big] \frac{M_x M_y}{M_0}
$$
and similarly for $\delta f_2$.


```python

```
