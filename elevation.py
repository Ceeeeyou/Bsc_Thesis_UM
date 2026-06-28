"""
Elevation module for vehicle-aware eco-routing.

Provides a uniform get_elevation(lat, lon) interface backed by:
  1. AHN3/AHN4 (Dutch national LIDAR DEM, ~0.5 m grid, ~5 cm vertical accuracy)
     — for any point in the Netherlands. Primary source.
  2. SRTM (~30 m global, ~7-10 m vertical accuracy) — fallback when AHN
     unavailable or for points outside NL.

Why AHN over SRTM for a Dutch thesis: NL is famously flat (~5-25 m relief
across Eindhoven), which puts elevation differences within SRTM's vertical
noise floor. AHN's centimeter-level accuracy is what makes the grade-work
signal meaningful here. AHN is also the standard data source any Dutch
transport-engineering examiner would expect.

ARCHITECTURE
============
On first use, the module downloads a single GeoTIFF tile spanning the bbox
of interest (cached in ~/.eco_routing/dem/), then queries it locally via
rasterio for fast lookups. The full AHN3 dataset is ~1 TB; we only fetch
the tiles we need.

DEPENDENCIES (run on your machine):
    pip install rasterio numpy requests

USAGE
=====
    from elevation import ElevationProvider
    dem = ElevationProvider(source='ahn3', cache_dir='~/.eco_routing/dem')
    # On first call, prepares cache for the bbox; subsequent calls are local
    dem.prepare_bbox(lat_min=51.40, lat_max=51.50, lon_min=5.35, lon_max=5.55)
    h = dem.get(51.4430, 5.4789)   # elevation in meters

For an OSMnx graph:
    add_elevation_to_graph(G, dem)
    # Adds 'elevation_m' to each node and 'dh_m' to each edge

REFERENCES
==========
- AHN documentation: https://www.ahn.nl/ (Dutch National Elevation File)
- PDOK service: https://www.pdok.nl/datasets (open data portal, including AHN3 download)
- SRTM v3: https://www2.jpl.nasa.gov/srtm/

CITATION (for thesis):
- Van der Sande, C., Soudarissanane, S., Khoshelham, K. (2010). Assessment of
  relative accuracy of AHN-2 laser scanning data using planar features.
  Sensors, 10(9), 8198-8214.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import math
import numpy as np


# =============================================================================
# Provider interface
# =============================================================================
@dataclass
class ElevationProvider:
    """Looks up elevation at a (lat, lon) coordinate.

    Caches a DEM raster locally on first prepare_bbox() call; subsequent
    get() calls are fast offline lookups.

    source: 'ahn3' (Dutch LIDAR, primary), 'srtm' (global, fallback),
            or 'flat' (always returns 0 — for testing without DEM).
    """
    source: str = 'ahn3'
    cache_dir: str = '~/.eco_routing/dem'
    _raster: Optional[object] = None     # rasterio dataset, populated lazily
    _band_data: Optional[np.ndarray] = None
    _transform: Optional[object] = None
    _bbox: Optional[tuple] = None

    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def prepare_bbox(self, lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float) -> None:
        """Ensure DEM data is available for the bbox; download if needed."""
        if self.source == 'flat':
            return                    # No-op; flat-earth provider

        elif self.source == 'srtm':
            self._prepare_srtm(lat_min, lat_max, lon_min, lon_max)

        elif self.source == 'ahn3':
            self._prepare_ahn3(lat_min, lat_max, lon_min, lon_max)

        else:
            raise ValueError(f"Unknown DEM source: {self.source!r}")

    # ------------------------------------------------------------------------
    def _prepare_ahn3(self, lat_min, lat_max, lon_min, lon_max):
        """Download AHN3 DTM 0.5 m tile via PDOK Atom feed.

        AHN3 tiles are organized in a grid of ~6.25 x 5 km cells named like
        '32fz1', '32gz2', etc. PDOK exposes them as direct downloads. The
        simplest reliable path is the predefined-grid endpoint:

            https://geodata.nationaalgeoregister.nl/ahn3/extract/ahn3_05m_dtm/<TILE>.zip

        We use the bbox to pick the covering tile(s), download as ZIP,
        extract the GeoTIFF inside.
        """
        try:
            import requests, zipfile, io
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.merge import merge
        except ImportError:
            raise RuntimeError(
                "AHN3 requires: pip install rasterio requests\n"
                "Or use source='srtm' for global SRTM, or 'flat' for testing."
            )

        # AHN3 tile naming: covering bbox via a published WCS endpoint.
        # PDOK exposes a Web Coverage Service at:
        #   https://service.pdok.nl/rws/ahn/wcs/v1_0
        # We can request a GeoTIFF for an arbitrary bbox in EPSG:28992 (RD).
        # Convert lat/lon (WGS84) to RD New (EPSG:28992) and issue a WCS GetCoverage.
        try:
            from pyproj import Transformer
        except ImportError:
            raise RuntimeError("AHN3 requires: pip install pyproj")

        # Pad the bbox a bit so we can answer queries near edges
        pad = 0.005
        lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad

        # Transform bbox corners to RD New
        wgs_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        rd_xs, rd_ys = [], []
        for la in (lat_min, lat_max):
            for lo in (lon_min, lon_max):
                x, y = wgs_to_rd.transform(lo, la)
                rd_xs.append(x); rd_ys.append(y)
        x_min, x_max = min(rd_xs), max(rd_xs)
        y_min, y_max = min(rd_ys), max(rd_ys)

        cache_path = Path(self.cache_dir) / (
            f"ahn3_dtm_{int(x_min)}_{int(y_min)}_{int(x_max)}_{int(y_max)}.tif"
        )
        if not cache_path.exists():
            print(f"  Downloading AHN DTM via PDOK WCS for "
                  f"RD bbox [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]...")

            try:
                from owslib.wcs import WebCoverageService
            except ImportError:
                raise RuntimeError(
                    "AHN download requires owslib. Run: py -m pip install owslib"
                )

            # PDOK's AHN WCS. Use WCS 1.0.0 (the service is most stable on it;
            # 2.0.1 SUBSET syntax returns 400). Request 5 m resolution so the
            # output stays under PDOK's 4000x4000-pixel cap — native data is
            # 0.5 m but 5 m is far finer than our ~30 m edge sampling needs.
            # PDOK caps output at 4000x4000 px. Pick a resolution that keeps
            # both dimensions under that, starting from 5 m and coarsening for
            # long inter-city corridors (e.g. Maasvlakte-Rotterdam ~40 km).
            # PDOK caps output at 4000x4000 px. The output pixel count on each
            # axis is span/res, so to keep BOTH axes under the cap we must pick
            # res from the LARGER span. Start at 5 m (far finer than our ~30 m
            # edge sampling) and coarsen as needed for long corridors.
            span_x = x_max - x_min
            span_y = y_max - y_min
            MAXPX = 3900   # safety margin under 4000
            res = max(5.0, span_x / MAXPX, span_y / MAXPX)
            res = round(res, 1)

            # Sanity guard: verify the resulting raster really is under the cap.
            # If a corridor is so long that even at this res a dimension still
            # exceeds the cap (shouldn't happen given the formula, but the WCS
            # also has an absolute area limit), coarsen further until it fits.
            while (span_x / res > MAXPX) or (span_y / res > MAXPX):
                res = round(res * 1.5, 1)

            # PDOK also rejects requests whose total area is too large
            # regardless of pixel count. Very long inter-city corridors
            # (e.g. Maasvlakte→city centre, ~40 km) can hit this even at
            # coarse resolution. Above ~25 km on either axis, AHN is the
            # wrong tool — relief over such a span is captured fine by SRTM,
            # and for flat NL corridors grade is negligible anyway. Signal
            # the caller to fall back rather than emit a broken request.
            MAX_SPAN_M = 25_000
            if span_x > MAX_SPAN_M or span_y > MAX_SPAN_M:
                raise RuntimeError(
                    f"AHN bbox too large for WCS: span "
                    f"{span_x/1000:.1f}x{span_y/1000:.1f} km exceeds "
                    f"{MAX_SPAN_M/1000:.0f} km. Use dem='srtm' or 'flat' for "
                    f"this corridor (long-corridor grade is better served by "
                    f"SRTM; AHN's 0.5 m precision is wasted at this scale)."
                )

            wcs = WebCoverageService(
                'https://service.pdok.nl/rws/ahn/wcs/v1_0',
                version='1.0.0',
                timeout=300,
            )
            response = wcs.getCoverage(
                identifier='dtm_05m',
                bbox=(x_min, y_min, x_max, y_max),
                format='GEOTIFF',
                crs='urn:ogc:def:crs:EPSG::28992',
                resx=res,
                resy=res,
            )
            content = response.read()

            # Validate: a failed WCS request returns an XML error, not a TIFF.
            # Real GeoTIFFs start with II* (little-endian) or MM* (big-endian).
            if not (content.startswith(b'II*') or content.startswith(b'MM*')):
                head = content[:400].decode('utf-8', errors='replace')
                raise RuntimeError(
                    "AHN WCS returned non-GeoTIFF data (likely an error):\n"
                    f"{head}\n"
                    "Common causes: bbox too large (PDOK caps at 4000x4000 px), "
                    "service down, or bad coverage id."
                )
            cache_path.write_bytes(content)
            print(f"  Saved: {cache_path}  ({len(content)/1024/1024:.1f} MB, {res:.1f}m)")

        # Open the cached GeoTIFF
        self._raster = rasterio.open(cache_path)
        self._band_data = self._raster.read(1)
        self._transform = self._raster.transform
        self._bbox = (lat_min, lat_max, lon_min, lon_max)
        self._wgs_to_rd = wgs_to_rd

    # ------------------------------------------------------------------------
    def _prepare_srtm(self, lat_min, lat_max, lon_min, lon_max):
        """Download SRTM GL1 (30 m) for the bbox via the OpenTopography REST
        API as a single GeoTIFF.

        This replaces the old `elevation` package, which (a) shells out to
        GDAL/make and is unreliable on Windows, and (b) shares its import
        name with THIS module (elevation.py), causing
        `import elevation` to import ourselves instead of the package.

        Requires a free OpenTopography API key (instant signup at
        https://opentopography.org → "Request an API key"). Provide it via:
          - the OPENTOPOGRAPHY_API_KEY environment variable, OR
          - a file named .opentopography.txt in the codes folder or your
            home directory, containing just the key.

        Global coverage (works for Germany, Netherlands, anywhere ±60° lat).
        For NL corridors prefer source='ahn3' (0.5 m). SRTM's ~30 m
        resolution and ~7 m vertical noise are fine where relief is large
        (e.g. Stuttgart's 150-235 m valley-to-hilltop climbs).
        """
        try:
            import requests
            import rasterio
        except ImportError:
            raise RuntimeError(
                "SRTM requires: pip install rasterio requests"
            )

        # Resolve the API key
        import os
        api_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
        if not api_key:
            for cand in (Path('.opentopography.txt'),
                         Path.home() / '.opentopography.txt'):
                if cand.exists():
                    api_key = cand.read_text().strip()
                    break
        if not api_key:
            raise RuntimeError(
                "SRTM via OpenTopography needs a free API key.\n"
                "  1. Sign up (free, instant): https://opentopography.org "
                "→ 'Request an API key'\n"
                "  2. Save it one of two ways:\n"
                "     - set env var:  setx OPENTOPOGRAPHY_API_KEY \"yourkey\"\n"
                "       (then open a NEW terminal)\n"
                "     - or create a file  .opentopography.txt  in the codes "
                "folder containing only the key\n"
                "Alternatively use --dem ahn3 (NL only) or --dem flat (no relief)."
            )

        cache_path = Path(self.cache_dir) / (
            f"srtm_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}.tif"
        )
        if not cache_path.exists():
            # Pad slightly so edge queries are covered
            pad = 0.005
            s, n = lat_min - pad, lat_max + pad
            w, e = lon_min - pad, lon_max + pad
            print(f"  Downloading SRTM GL1 30m via OpenTopography for bbox "
                  f"[{s:.4f}, {w:.4f}, {n:.4f}, {e:.4f}]...")
            url = (
                "https://portal.opentopography.org/API/globaldem"
                "?demtype=SRTMGL1"
                f"&south={s:.6f}&north={n:.6f}&west={w:.6f}&east={e:.6f}"
                "&outputFormat=GTiff"
                f"&API_Key={api_key}"
            )
            r = requests.get(url, timeout=300)
            if r.status_code != 200:
                raise RuntimeError(
                    f"OpenTopography returned HTTP {r.status_code}.\n"
                    f"First 300 chars: {r.text[:300]}\n"
                    "Common causes: bad/expired API key, bbox too large, "
                    "or service outage."
                )
            content = r.content
            # Validate GeoTIFF magic bytes (II* little-endian / MM* big-endian)
            if not (content.startswith(b'II*') or content.startswith(b'MM*')):
                head = content[:300].decode('utf-8', errors='replace')
                raise RuntimeError(
                    "OpenTopography did not return a GeoTIFF (likely an error "
                    f"message):\n{head}"
                )
            cache_path.write_bytes(content)
            print(f"  Saved: {cache_path}  ({len(content)/1024/1024:.1f} MB)")

        self._raster = rasterio.open(cache_path)
        self._band_data = self._raster.read(1)
        self._transform = self._raster.transform
        self._bbox = (lat_min, lat_max, lon_min, lon_max)

    # ------------------------------------------------------------------------
    def get(self, lat: float, lon: float) -> float:
        """Look up elevation at (lat, lon) in meters above sea level (NAP/MSL).

        Returns 0.0 if source is 'flat'. Returns NaN if the point is outside
        the prepared bbox.
        """
        if self.source == 'flat':
            return 0.0
        if self._raster is None:
            raise RuntimeError("Call prepare_bbox() before get()")

        # Convert to raster CRS if needed
        if self.source == 'ahn3':
            x, y = self._wgs_to_rd.transform(lon, lat)
        else:
            x, y = lon, lat

        # Convert (x, y) in raster CRS to pixel (row, col)
        row, col = self._raster.index(x, y)
        if not (0 <= row < self._band_data.shape[0]
                and 0 <= col < self._band_data.shape[1]):
            return float('nan')

        val = float(self._band_data[row, col])

        # AHN3 nodata is typically 3.4e38 or -32768; SRTM is -32768
        if val < -1000 or val > 9000:
            return float('nan')
        return val

    def get_many(self, lats, lons):
        """Vectorized version of get()."""
        return np.array([self.get(la, lo) for la, lo in zip(lats, lons)])


# =============================================================================
# Graph annotation
# =============================================================================
def add_elevation_to_graph(G, dem: ElevationProvider,
                            sample_spacing_m: float = 30.0,
                            max_samples_per_edge: int = 50) -> None:
    """Annotate each node with elevation and each edge with grade-work-relevant
    fields, sampling elevation at multiple points along the edge geometry.

    Per-edge fields written:
        dh_m          endpoint-to-endpoint elevation difference (signed)
        dh_pos_m      sum of positive elevation gains along the edge
                      (= integral of max(0, dh) over sub-segments; this is
                      the ICE-correct quantity for grade work since downhill
                      doesn't recover energy per ISO 23795-1 B.4)
        elev_profile  list of (cumulative_distance_m, elevation_m) samples
                      (kept for diagnostics / visualization)

    Why dh_pos_m matters: for a 2 km edge that climbs 8 m and descends 8 m,
    the endpoint dh_m is 0 but dh_pos_m is ~8 m. The ICE truck pays grade
    work for the 8 m climb; the descent is brake-dissipated. Without
    multi-point sampling, the model would under-estimate grade work on
    rolling terrain.

    Parameters
    ----------
    sample_spacing_m : target distance between elevation samples along an
                       edge. ~30 m is a sweet spot — fine enough to catch
                       overpasses (typical length 20-50 m), coarse enough
                       to avoid integrating DEM noise. Document this in
                       the thesis as a design parameter.
    max_samples_per_edge : safety cap for very long edges.
    """
    # Node elevations
    for n, data in G.nodes(data=True):
        lat = data.get('y'); lon = data.get('x')
        if lat is None or lon is None:
            data['elevation_m'] = float('nan')
            continue
        data['elevation_m'] = dem.get(lat, lon)

    # Edge multi-point sampling
    missing_count = 0
    total_dh_pos_endpoint = 0.0
    total_dh_pos_multipoint = 0.0
    for u, v, k, data in G.edges(keys=True, data=True):
        h_u = G.nodes[u].get('elevation_m', float('nan'))
        h_v = G.nodes[v].get('elevation_m', float('nan'))
        length_m = float(data.get('length', 0.0))

        if math.isnan(h_u) or math.isnan(h_v):
            data['dh_m'] = 0.0
            data['dh_pos_m'] = 0.0
            data['elev_profile'] = []
            missing_count += 1
            continue

        # Endpoint-only result (kept for back-compat and as a diagnostic)
        data['dh_m'] = h_v - h_u

        # Multi-point sampling: take edge `geometry` if OSMnx attached one
        # (a shapely LineString), otherwise straight-line interpolate.
        sample_points = _interpolate_edge(
            G, u, v, data, length_m,
            sample_spacing_m, max_samples_per_edge,
        )

        # Look up elevation at each sample point
        elevations = []
        for lat_i, lon_i, d_i in sample_points:
            h = dem.get(lat_i, lon_i)
            if math.isnan(h):
                # Fall back to linear interpolation between endpoints
                h = h_u + (h_v - h_u) * (d_i / length_m if length_m > 0 else 0.5)
            elevations.append((d_i, h))

        data['elev_profile'] = elevations

        # Sum positive elevation gains across sub-segments
        dh_pos = 0.0
        for (d1, h1), (d2, h2) in zip(elevations[:-1], elevations[1:]):
            seg_dh = h2 - h1
            if seg_dh > 0:
                dh_pos += seg_dh
        data['dh_pos_m'] = dh_pos

        total_dh_pos_endpoint += max(0, h_v - h_u)
        total_dh_pos_multipoint += dh_pos

    if missing_count:
        print(f"  Note: {missing_count} edges had missing elevation, set dh_m=0")

    if total_dh_pos_endpoint > 0:
        ratio = total_dh_pos_multipoint / total_dh_pos_endpoint
        print(f"  Multi-point grade gain: {total_dh_pos_multipoint:.0f} m "
              f"(endpoint-only: {total_dh_pos_endpoint:.0f} m, "
              f"ratio {ratio:.2f}× — higher means more hidden relief)")


def _interpolate_edge(G, u, v, edge_data, length_m: float,
                        spacing_m: float, max_samples: int) -> list:
    """Return [(lat, lon, cumulative_distance_m), ...] sampled along the edge.

    Uses the OSMnx `geometry` (shapely LineString) if present, otherwise
    falls back to a straight line between node endpoints. Always includes
    the two endpoints.
    """
    # Number of samples to take
    n_samples = max(2, min(max_samples, int(length_m / spacing_m) + 1))

    geom = edge_data.get('geometry')
    if geom is not None:
        # OSMnx attached a shapely LineString with the curved road geometry
        try:
            # Coordinates are typically (lon, lat) for OSMnx geometries
            coords = list(geom.coords)
            # Length-parameterize and sample
            return _sample_linestring(coords, n_samples, length_m)
        except Exception:
            pass    # fall through to straight-line

    # Fallback: straight-line interpolation between nodes
    lon_u, lat_u = G.nodes[u].get('x'), G.nodes[u].get('y')
    lon_v, lat_v = G.nodes[v].get('x'), G.nodes[v].get('y')
    points = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        lat = lat_u + t * (lat_v - lat_u)
        lon = lon_u + t * (lon_v - lon_u)
        d = t * length_m
        points.append((lat, lon, d))
    return points


def _sample_linestring(coords: list, n_samples: int, total_length_m: float) -> list:
    """Sample n_samples points evenly along a (lon, lat) LineString,
    returning (lat, lon, cumulative_distance_m). Distance is approximate
    (uses small-angle haversine since edges are short)."""
    import math as _math

    # Cumulative length along the linestring (haversine)
    cum = [0.0]
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        cum.append(cum[-1] + _haversine_m(lat1, lon1, lat2, lon2))

    total_geom_len = cum[-1] if cum[-1] > 0 else total_length_m

    out = []
    for i in range(n_samples):
        target = (i / (n_samples - 1)) * total_geom_len
        # Find the LineString segment containing this distance
        for j in range(len(cum) - 1):
            if cum[j] <= target <= cum[j + 1]:
                seg_len = cum[j + 1] - cum[j]
                frac = (target - cum[j]) / seg_len if seg_len > 0 else 0
                lon = coords[j][0] + frac * (coords[j + 1][0] - coords[j][0])
                lat = coords[j][1] + frac * (coords[j + 1][1] - coords[j][1])
                # Re-scale distance to caller's notion of edge length
                d_scaled = target * (total_length_m / total_geom_len) if total_geom_len > 0 else 0
                out.append((lat, lon, d_scaled))
                break
        else:
            # Past the end (numerical edge case): clamp
            lon, lat = coords[-1]
            out.append((lat, lon, total_length_m))
    return out


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two (lat, lon) points."""
    import math as _math
    R = 6371000
    p1 = _math.radians(lat1); p2 = _math.radians(lat2)
    dp = _math.radians(lat2 - lat1); dl = _math.radians(lon2 - lon1)
    a = _math.sin(dp/2)**2 + _math.cos(p1) * _math.cos(p2) * _math.sin(dl/2)**2
    return 2 * R * _math.asin(_math.sqrt(a))


# =============================================================================
# Synthetic provider for testing without internet
# =============================================================================
class SyntheticElevationProvider(ElevationProvider):
    """Returns a deterministic elevation surface for testing. Models the
    Eindhoven area with a gentle south-rising slope (Nuenen low to Aalst-Waalre
    higher), plus a few local features (motorway overpasses, Dommel valley)."""

    def __init__(self):
        super().__init__(source='flat')   # bypass real DEM machinery
        self.source = 'synthetic'

    def prepare_bbox(self, *args, **kwargs):
        pass    # no-op

    def get(self, lat: float, lon: float) -> float:
        # Base: gentle elevation gradient, low north and south, ridge in middle
        # (Eindhoven area: ~17 m typical, locally up to ~30 m on sandy ridges)
        base = 17.0 + 8.0 * math.exp(-((lat - 51.42) ** 2) / 0.001)

        # Local features:
        # - The A2 motorway overpasses (around lat 51.46, lon 5.43-5.45):
        #   raised by ~5 m
        if 51.455 < lat < 51.465 and 5.43 < lon < 5.46:
            base += 5.0
        # - The Dommel river valley (lat ~ 51.44, lon 5.47-5.50): slightly lower
        if 51.44 < lat < 51.45 and 5.47 < lon < 5.50:
            base -= 2.0
        # - South Eindhoven ridge (~51.39): higher
        if 51.38 < lat < 51.40:
            base += 6.0
        return base


# =============================================================================
# CLI / smoke test
# =============================================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Elevation lookup smoke test")
    ap.add_argument('--source', default='synthetic',
                    choices=['ahn3', 'srtm', 'flat', 'synthetic'])
    ap.add_argument('--lat', type=float, default=51.4430)
    ap.add_argument('--lon', type=float, default=5.4789)
    args = ap.parse_args()

    if args.source == 'synthetic':
        dem = SyntheticElevationProvider()
    else:
        dem = ElevationProvider(source=args.source)
        dem.prepare_bbox(args.lat - 0.02, args.lat + 0.02,
                          args.lon - 0.02, args.lon + 0.02)
    print(f"Elevation at ({args.lat}, {args.lon}): {dem.get(args.lat, args.lon):.2f} m")
