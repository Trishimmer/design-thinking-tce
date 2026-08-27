import os
import uuid
import numpy as np

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for Flask's threaded dev server
import matplotlib.pyplot as plt

try:
    import rasterio
except Exception:
    rasterio = None


def compute_ndvi_from_geotiff(path, out_dir='static'):
    if rasterio is None:
        raise RuntimeError('rasterio is required to process GeoTIFFs')

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(path) as src:
        # Prefer sentinel-like band ordering: 3=red,4=nir when available
        if src.count >= 4:
            red = src.read(3).astype('float32')
            nir = src.read(4).astype('float32')
        elif src.count >= 2:
            # Fallback: assume band1=red, band2=nir
            red = src.read(1).astype('float32')
            nir = src.read(2).astype('float32')
        else:
            raise ValueError('GeoTIFF must contain at least Red and NIR bands (2 bands).')

        # Compute NDVI
        np.seterr(divide='ignore', invalid='ignore')
        denom = (nir + red)
        ndvi = (nir - red) / denom
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

        mean = float(np.mean(ndvi))
        std = float(np.std(ndvi))

        # Create a visualisation and save it
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fname = f"ndvi_{uuid.uuid4().hex}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        return {'mean': mean, 'std': std, 'image': out_path}