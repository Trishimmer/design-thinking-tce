#!/usr/bin/env python3
"""
Generate a small synthetic GeoTIFF with Red and NIR bands for testing the app's upload/NDVI flow.

Usage:
  python scripts/generate_sample_geotiff.py output.tif

The produced file will have 4 bands where band 3 is Red and band 4 is NIR (common Sentinel ordering).
"""
import sys
import numpy as np
try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:
    rasterio = None


def generate(path='sample_geotiff.tif', width=256, height=256):
    if rasterio is None:
        raise RuntimeError('rasterio is required to generate a GeoTIFF. Install rasterio first.')

    # Create a simple gradient for red and a stronger pattern for NIR
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)

    # Base red reflectance as gradient
    red = (xv * 3000).astype('uint16')

    # NIR stronger in the centre to simulate vegetation
    nir = ((xv + yv) * 1500 + (1 - np.hypot(xv - 0.5, yv - 0.5)) * 2000).clip(0, 65535).astype('uint16')

    # Additional bands to make a 4-band GeoTIFF (green, blue placeholders)
    green = (red * 0.8).astype('uint16')
    blue = (red * 0.6).astype('uint16')

    transform = from_origin(0.0, 0.0, 1.0, 1.0)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 4,
        'dtype': 'uint16',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw'
    }

    with rasterio.open(path, 'w', **profile) as dst:
        # Write bands: 1=green,2=blue,3=red,4=nir to match our app's expected red=3,nir=4
        dst.write(green, 1)
        dst.write(blue, 2)
        dst.write(red, 3)
        dst.write(nir, 4)

    print(f'Generated sample GeoTIFF: {path}')


if __name__ == '__main__':
    out = 'sample_geotiff.tif'
    if len(sys.argv) > 1:
        out = sys.argv[1]
    try:
        generate(out)
    except Exception as e:
        print('Error:', e)
        print('Install rasterio (and its GDAL dependencies) to run this script.')