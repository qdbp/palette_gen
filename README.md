
### WIP CODEBASE

## Description


This repo contains code to generate pleasing
[HK effect](https://en.wikipedia.org/wiki/Helmholtz%E2%80%93Kohlrausch_effect)
-corrected color palettes with distinguishable colors. The main aim is to aid in
color scheme design.


### Description

WIP


### Examples




### Bonus: fast color converters

In order to create these paletters, this repo contains a number of `@njit`
-decorated vectorized color conversion functions. They are very, very fast and
are a great choice where the more OO-style converters from libraries
like `colour-science` or `colour-math`
are unsuitable. These functions are tested to reproduce identical results to
those in
`colour-science`.

```
HSV_to_RGB_jit(__colmat_hsv): 7.74e-08 seconds per call per color for 10000 rows
sRGB_to_XYZ_jit(__colmat_rgb): 6.02e-08 seconds per call per color for 10000 rows
xyY_to_XYZ_jit(__colmat_xyy): 4.77e-09 seconds per call per color for 10000 rows
XYZ_to_xyY_D65_jit(__colmat_xyz): 2.18e-08 seconds per call per color for 10000 rows
XYZ_to_Luv_D65_jit(__colmat_xyz): 2.38e-08 seconds per call per color for 10000 rows
XYZ_to_Lab_D65_jit(__colmat_xyz): 5.22e-08 seconds per call per color for 10000 rows
Luv_to_LCHuv_jit(__colmat_luv): 3.01e-08 seconds per call per color for 10000 rows
Lab_to_LCHab_jit(__colmat_luv): 3.01e-08 seconds per call per color for 10000 rows
dE_2000_jit(__colmat_luv[:-1], __colmat_luv[1:]): 1.74e-07 seconds per call per color for 9999 rows

```

### References
[1]: Kaiser, P. K., CIE Journal 5, 57 (1986)
[2]: Donofrio, R. L. (2011). Review Paper: The Helmholtz-Kohlrausch effect. Journal of the Society for Information Display, 19(10), 658. doi:10.1889/jsid19.10.658