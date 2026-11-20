"""Procedurally render the illustrative space images used by the dashboard (artist's-style renders, NOT telescope photographs).

Run:  python scripts/make_space_images.py      (needs numpy, scipy, pillow; output -> assets/images/)
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parent.parent / "assets" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def fbm(shape, rng, octaves=6, base=4.0, persistence=0.55):
    """Fractal noise in [0, 1] built from smoothed random fields."""
    h, w = shape
    total = np.zeros(shape)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        sigma = max(h, w) / (base * 2 ** o)
        f = gaussian_filter(rng.standard_normal(shape), sigma=max(sigma, 0.6), mode="wrap")
        f = (f - f.mean()) / (f.std() + 1e-9)
        total += amp * f
        norm += amp
        amp *= persistence
    total /= norm
    return np.clip(0.5 + total * 0.5, 0, 1)


def stars(shape, rng, density=0.0016, glow=True):
    h, w = shape
    img = np.zeros((h, w, 3))
    n = int(h * w * density)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    mag = rng.power(0.35, n)
    tint = rng.choice(3, n, p=[.6, .25, .15])
    cols = np.array([[1, 1, 1], [.75, .85, 1], [1, .88, .7]])[tint]
    img[ys, xs] = cols * mag[:, None]
    soft = gaussian_filter(img, sigma=(0.8, 0.8, 0))
    out = soft * 2.2
    if glow:
        bright = mag > 0.93
        big = np.zeros_like(img)
        big[ys[bright], xs[bright]] = cols[bright]
        out += gaussian_filter(big, sigma=(3.2, 3.2, 0)) * 9
    return out


def milky_way(shape, rng):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.deg2rad(-18)
    d = (yy - h / 2) * np.cos(ang) - (xx - w / 2) * np.sin(ang)
    band = np.exp(-(d / (h * 0.22)) ** 2)
    n1, n2, n3 = fbm(shape, rng, 6, 3), fbm(shape, rng, 6, 5), fbm(shape, rng, 5, 8)
    dust = np.clip(n2 * 1.6 - 0.55, 0, 1)
    glow = band * (0.35 + 0.9 * n1) * (1 - 0.75 * dust * band)
    col = np.stack([glow * (0.55 + 0.35 * n3), glow * (0.5 + 0.2 * n1), glow * (0.85)], -1)
    return col * 0.55


def sphere(size, rng, kind, light=(-0.55, -0.35, 0.75), rot=0.0):
    """Shaded planet with surface texture and an atmospheric limb. Returns (rgb, alpha) on a size x size canvas
    where the planet radius is 0.62*size/2... glow extends beyond."""
    R = 0.70
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.hypot(x, y)
    inside = r <= R
    nx, ny = x / R, y / R
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    L = np.array(light) / np.linalg.norm(light)
    ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    shade = ndl ** 0.85
    tex = fbm((size, size), rng, 7, 3)
    lat = ny
    if kind == "gas":
        warp = (fbm((size, size), rng, 5, 6) - 0.5) * 0.35
        bands = 0.5 + 0.5 * np.sin((lat + warp) * 17 + 2.5 * tex)
        stripes = gaussian_filter(bands, 1.2)
        c1, c2, c3 = np.array([.93, .74, .48]), np.array([.62, .36, .22]), np.array([.98, .90, .75])
        albedo = (c1 * (1 - stripes[..., None]) + c2 * stripes[..., None]) * (0.8 + 0.4 * tex[..., None])
        albedo = albedo * 0.9 + c3 * (np.clip((tex[..., None] - 0.6) * 3, 0, 1)) * 0.18
        atm = np.array([1.0, .75, .45])
    elif kind == "ocean":
        land = fbm((size, size), rng, 7, 2.6)
        cont = land > 0.56
        ocean = np.array([.05, .20, .48]) + 0.10 * tex[..., None]
        ground = np.array([.22, .34, .16]) * (0.7 + 0.6 * tex[..., None]) + np.array([.25, .2, .1]) * (land[..., None] > 0.64)
        albedo = np.where(cont[..., None], ground, ocean)
        clouds = np.clip((fbm((size, size), rng, 7, 3.5) - 0.50) * 3.2, 0, 1)
        albedo = albedo * (1 - clouds[..., None]) + np.array([.96, .97, 1.0]) * clouds[..., None]
        polar = np.clip((np.abs(ny) - 0.82) * 6, 0, 1)
        albedo = albedo * (1 - polar[..., None]) + polar[..., None] * 0.95
        atm = np.array([.35, .62, 1.0])
    else:  # ice / rocky super-earth
        crater = fbm((size, size), rng, 7, 5)
        albedo = (np.array([.52, .66, .82]) * (0.55 + 0.75 * tex[..., None]) + np.array([.25, .27, .35]) * np.clip((crater[..., None] - 0.55) * 3, 0, 1))
        frost = np.clip((fbm((size, size), rng, 6, 4) - 0.5) * 3, 0, 1)
        albedo = albedo * (1 - 0.6 * frost[..., None]) + 0.6 * frost[..., None]
        atm = np.array([.6, .85, 1.0])
    surf = albedo * (0.04 + 1.05 * shade[..., None])
    spec = (np.clip(2 * nz * (nx * L[0] + ny * L[1] + nz * L[2]) - L[2], 0, 1) ** 40) * (0.35 if kind == "ocean" else 0.05)
    surf = surf + spec[..., None]
    fres = (1 - nz) ** 2.6
    surf = surf + (fres * (0.15 + 1.6 * shade))[..., None] * atm * 0.9
    rgb = np.zeros((size, size, 3))
    rgb[inside] = surf[inside]
    # outer glow (atmospheric scattering beyond the limb)
    gl = np.exp(-np.clip(r - R, 0, None) * 26) * (r > R - 0.005)
    lit_side = np.clip((x * L[0] + y * L[1]) / (r + 1e-6) * 0.5 + 0.55, 0.1, 1)
    glow = gl[..., None] * atm * (0.5 * lit_side[..., None])
    rgb = np.where(inside[..., None], rgb, glow)
    alpha = np.where(inside, 1.0, np.clip(gl * 0.9, 0, 1))
    return np.clip(rgb, 0, 1.6), alpha


def tonemap(a):
    a = np.clip(a, 0, None)
    a = 1 - np.exp(-a * 1.15)
    return np.clip(a ** (1 / 1.05), 0, 1)


def save(arr, name, q=90):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(OUT / name, quality=q, optimize=True)
    print("wrote", name)


def planet_image(kind, name, seed, size=640, ring=False):
    rng = np.random.default_rng(seed)
    bg = stars((size, size), rng, 0.0012) + milky_way((size, size), rng) * 0.35
    p, a = sphere(size, rng, kind)
    comp = bg * (1 - a[..., None]) + p * a[..., None] + p * (1 - a[..., None]) * 0
    save(tonemap(comp), name)


def nebula_image():
    rng = np.random.default_rng(5)
    h, w = 560, 1100
    n1, n2, n3, n4 = (fbm((h, w), rng, 7, b) for b in (6, 8, 10, 14))
    dens = np.clip((n1 - 0.40) * 2.6, 0, 1) ** 1.2
    violet, teal, coral = np.array([.45, .30, 1.0]), np.array([.15, .85, 1.0]), np.array([1.0, .38, .45])
    col = (dens[..., None] * (violet * n2[..., None] + teal * (1 - n2[..., None])) * 0.9
           + np.clip(n3 - 0.55, 0, 1)[..., None] * 3.0 * coral * dens[..., None])
    filaments = np.clip(1 - np.abs(n4 - 0.5) * 9, 0, 1)[..., None] * dens[..., None] * np.array([.7, .9, 1.0]) * 0.6
    dark = np.clip((fbm((h, w), rng, 6, 3.5) - 0.58) * 4, 0, 1)
    col = (col + filaments) * (1 - 0.65 * dark[..., None])
    img = col * 1.1 + stars((h, w), rng, 0.0022)
    save(tonemap(img), "nebula.jpg")


def transit_image():
    rng = np.random.default_rng(9)
    h, w = 560, 1100
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy, Rs = w * 0.5, h * 0.5, 210.0
    r = np.hypot(xx - cx, yy - cy) / Rs
    mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))
    limb = 1 - 0.62 * (1 - mu) - 0.2 * (1 - mu) ** 2
    gran = fbm((h, w), rng, 7, 22)
    surface = limb * (0.8 + 0.4 * gran)
    star_col = np.array([1.0, .74, .36])
    disk = (r <= 1)[..., None] * surface[..., None] * star_col * 1.25
    corona = (np.exp(-np.clip(r - 1, 0, None) * 3.2) * (r > 1))[..., None] * np.array([1.0, .5, .22]) * 0.5
    img = disk + corona
    px, py, Rp = cx + 70, cy - 12, 34.0
    rp = np.hypot(xx - px, yy - py)
    planet = rp <= Rp
    halo = np.exp(-np.clip(rp - Rp, 0, None) / 5.5) * (rp > Rp)
    img = np.where(planet[..., None], np.array([.01, .01, .02]), img)
    img = img + (halo * np.clip(1 - r, 0, 1) * 0.0)[..., None]
    rim = (halo * 0.5)[..., None] * np.array([.5, .75, 1.0]) * (r > 1)[..., None]
    img = img + rim
    sky = stars((h, w), rng, 0.0016) * (r > 1.0)[..., None] * np.clip(r - 1.0, 0, 1)[..., None] ** 0.4
    img = img + sky
    save(tonemap(img), "transit.jpg")


if __name__ == "__main__":
    planet_image("gas", "gas_giant.jpg", 3)
    planet_image("ocean", "ocean_world.jpg", 12)
    planet_image("ice", "ice_world.jpg", 7)
    nebula_image()
    transit_image()
