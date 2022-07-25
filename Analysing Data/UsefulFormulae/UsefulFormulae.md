Unless said otherwise, assume all units used in the equations below require values in SI units! 

## Useful Constants

| Name | Symbol | Value | Unit |
| --- | --- | --- | --- |
| Speed of light | $c$ | $299792458$ | m/s |
| Meters to parsec conversion | N/A | $3.086 \times 10^{16}$ | m/pc |
| Solar mass | $M_\odot$ | $1.98 \times 10^{30}$ | kg |
| Solar luminosity | $L_\odot$ | $3.828 \times 10^{26}$ | W |
| Solar absolute magntiude (bolometric) | $M_\odot$ | 4.83 | N/A
| Planck's constant | $h$ | $6.626 \times 10^{-34}$ | J.s |
| Boltzmann's constant | $k_B$ | $1.38 \times 10^{-23}$ | m$^2$.kg.s$^{-2}$.K$^{-1}$ |
| Stefan-Boltzmann constant | $\sigma$ | $5.67 \times 10^{-8}$ | W.m$^{-2}$.K$^{-4}$ |

## General Formulae

For some equatorial angle $\theta$ and polar angle $\phi$, generating cartesian coordinates from spherical coordinates is done by

$$ x =  r \cos (\theta) \sin (\phi) $$

$$ y = r \sin (\theta) \sin (\phi) $$

$$ z = r \cos (\phi) $$

To find the distance to an object from a parallax measurement, $p$ (with their angles in units of arcseconds!), 

$$ d = \frac{1}{p"} \text{ pc} $$

## Stars

### Luminosity and Flux

The flux and luminosity of a star are related when considering their distance from an observer by 

$$ F = \frac{L}{4 \pi r^2} $$

The _bolometric_ luminosity of a star is related to its radius and temperature via the Stefan-Boltzmann law, 

$$ L_\text{bol} = 4 \pi R^2 \sigma T^4 $$

While not explicitly required at all in analysis of this data, you may choose to work with fluxes in terms of the _magnitude_ scale. The connection between distance, $d$, apparent magnitude $m$, and absolute magnitude, $M$ of a star is given by 

$$ d = 10^{(m - M + 5) / 5} \text{ pc} $$

The luminosity ratio of two stars is then 

$$ \frac{L_2}{L_1} = 100^{(M_1 - M_2) / 5} $$

and so, in terms of solar luminosities and magnitude, 

$$ M = M_\odot - 2.5 \log_{10} \left(\frac{L}{L_\odot}\right) $$ 

### Variability

The variability of stars follows a consistent wave (that is to say, it's periodicity doesn't change over time). Most waveforms can be approximated as sinusoidal waves, with function

$$ f(x) = A \sin \left(\frac{2\pi}{P}(x - S)\right) + E $$

While not all waves might follow a nice sine function, their periodicity can usually be approximated as one so it's a useful tool!

### Spectra
The spectrum of stars follows (pretty closely) a Planck function for a blackbody. This is given below, for a wavelength $\lambda$ (meters), temperature $T$ (K), and constants $h$ (Planck's constant), $c$ (speed of light), and $k_B$ (Boltzmann's constant).

$$ B_\lambda (\lambda, T) = \frac{2hc^2}{\lambda^5} \frac{1}{\text{exp}\left(\frac{hc}{\lambda k_B T}\right) - 1} $$

This on its own isn't very helpful, but you can use it to find the _monochromatic_ luminosity of a star (its luminosity in only one colour).

$$  L_\lambda = 4\pi^2 R^2 B_\lambda $$

The location of the peak of the blackbody curve for a star is related to its temperature by

$$ \lambda_\text{max} T \approx 0.29 \text{ cm K} $$

Source: [http://burro.cwru.edu/academics/Astr221/Light/blackbody.html](http://burro.cwru.edu/academics/Astr221/Light/blackbody.html)

## Black Holes

The (Schwarzschild) radius of a black hole is related to its mass by 

$$ r_S = \frac{2GM}{c^2} $$

The eddington luminosity of a black hole is approximated by 

$$L_\text{edd} \approx 3 \times 10^4 M$$

where $M$ is in terms of the solar masses of the black hole, and the output value of $L_\text{edd}$ is in terms of solar luminosities.

## Galaxies

### Rotational Velocities

The magnitude of the orbital velocity of a star about the center of a galaxy can be approximated by

$$ v = \sqrt{\frac{G M(<R)}{R}} $$

where $M(<R)$ represents _all_ of the mass enclosed within the orbital radius of the star, and so $R$ is then obviously the orbital radius of the star. $G$ here is the gravitational constant. That mass term may not include _only_ visible matter, and some dark matter may be present too. In the case of dark matter (you might be able to infer its presence by the shape of the rotation curve!), the Navarro-Frenk-White (NFW) profile of dark matter (which models an isotropic but density-changing mass distribution of dark matter) must be accounted for. 


## Clusters

The [above section](#rotational-velocities) for rotational velocities holds in the context of galaxy clusters too, practically identically (except instead of dealing with stars, we're dealing with galaxies as a whole). 


## Universe

The velocity of distant objects due to Hubble recession is related to Hubble's constant ($H_0$) and their distance away ($D$) by 

$$ v = H_0 D $$

Pay careful attention to unit conversions with this formula! 

