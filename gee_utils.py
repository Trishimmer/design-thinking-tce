import os
import io
import tempfile
import requests
import zipfile
import json
import rasterio


try:
    import ee
except Exception:
    ee = None


def initialize_ee(project=None):
    if ee is None:
        raise RuntimeError('earthengine-api is not installed')

    if project is None:
        project = os.environ.get('GEE_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT') or 'crop-recommendation-506706'

    last_err = None

    # 1) Straightforward init
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        return
    except Exception as e:
        last_err = e

    # 2) Service account credentials
    gcred = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if gcred and os.path.exists(gcred):
        try:
            with open(gcred, 'r') as fh:
                j = json.load(fh)
            service_account = j.get('client_email')
            if service_account:
                creds = ee.ServiceAccountCredentials(service_account, gcred)
                ee.Initialize(credentials=creds, project=project) if project else ee.Initialize(credentials=creds)
                return
        except Exception as e:
            last_err = e

    # 3) Interactive auth, same project preserved
    try:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
        return
    except Exception as e:
        last_err = e

    raise RuntimeError(
        'Google Earth Engine initialization failed. Run `earthengine authenticate` or set service account credentials. '
        'ee.Initialize: ' + str(last_err)
    )


import rasterio

def fetch_sentinel2_red_nir(lat, lon, start_date=None, end_date=None, buffer_m=500, out_dir='static'):
    """Fetch Sentinel-2 median Red+NIR GeoTIFF around a point using Earth Engine.

    Returns dict with keys: 'path' (local tif path)
    """
    if ee is None:
        raise RuntimeError('earthengine-api is not installed')

    initialize_ee()

    if start_date is None or end_date is None:
        import datetime
        end = datetime.date.today()
        start = end - datetime.timedelta(days=30)
        start_date = start.isoformat()
        end_date = end.isoformat()

    point = ee.Geometry.Point([float(lon), float(lat)])
    region = point.buffer(buffer_m).bounds()

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )

    image = collection.median().select(['B4', 'B8'])

    region_info = region.getInfo()
    region_geojson = json.dumps(region_info)
    url = image.getDownloadURL({'scale': 10, 'crs': 'EPSG:4326', 'region': region_geojson, 'fileFormat': 'GeoTIFF'})

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.close()

    local_path = None
    try:
        with zipfile.ZipFile(tmp.name, 'r') as z:
            # Extract every band tif, keyed by band name (B4, B8, ...)
            band_files = {}
            for name in z.namelist():
                if name.lower().endswith(('.tif', '.tiff')):
                    extract_path = os.path.join(out_dir, os.path.basename(name))
                    with open(extract_path, 'wb') as f:
                        f.write(z.read(name))
                    for band in ('B4', 'B8'):
                        if band in name:
                            band_files[band] = extract_path

            if 'B4' not in band_files or 'B8' not in band_files:
                raise RuntimeError(f'Expected B4 and B8 tif files in zip, found: {list(band_files.keys())}')

            # Stack B4 (Red) and B8 (NIR) into a single 2-band GeoTIFF
            with rasterio.open(band_files['B4']) as red_src:
                red_data = red_src.read(1)
                profile = red_src.profile.copy()
            with rasterio.open(band_files['B8']) as nir_src:
                nir_data = nir_src.read(1)

            profile.update(count=2)
            merged_path = os.path.join(out_dir, 'download_red_nir.tif')
            with rasterio.open(merged_path, 'w', **profile) as dst:
                dst.write(red_data, 1)
                dst.write(nir_data, 2)

            local_path = merged_path

            # Clean up the individual single-band files
            for f in band_files.values():
                try:
                    os.remove(f)
                except Exception:
                    pass

    except zipfile.BadZipFile:
        out_path = os.path.join(out_dir, os.path.basename(tmp.name) + '.tif')
        os.rename(tmp.name, out_path)
        local_path = out_path

    try:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
    except Exception:
        pass

    if local_path is None:
        raise RuntimeError('Failed to obtain GeoTIFF from Earth Engine download')

    return {'path': local_path}


def get_service_account_email():
    """Return client_email from GOOGLE_APPLICATION_CREDENTIALS if present, else None."""
    gcred = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if gcred and os.path.exists(gcred):
        try:
            with open(gcred, 'r') as fh:
                j = json.load(fh)
            return j.get('client_email')
        except Exception:
            return None
    return None

