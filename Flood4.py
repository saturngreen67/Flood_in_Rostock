import os
import time
import warnings
import requests
import numpy as np
import osmnx as ox
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from scipy.ndimage import (zoom, binary_dilation, distance_transform_edt,
                           label, minimum_filter)
from scipy.stats.qmc import LatinHypercube, scale as lhs_scale
from landlab import RasterModelGrid
from landlab.components import OverlandFlow

warnings.filterwarnings('ignore')

N_SCENARIOS = 2000
N_WORKERS   = 21
OUTPUT_DIR  = "scenarios"

CONFIG = {
    "RESOLUTION":     3.0,
    "NDVI_THRESHOLD": 0.6,
    "IMP_URBAN":      0.80,
    "IMP_STREET":     0.95,
    "IMP_BUILDING":   1.00,
    "IMP_EXIST_GI":   0.10,
    "N_DEFAULT":      0.035,
    "N_STREET":       0.015,
    "N_GREEN":        0.060,
    "N_BUILDING":     0.150,
    "N_WATER":        0.025,
    "INTENSITY_MIN":  2.0,   "INTENSITY_MAX":  50.0,
    "DURATION_MIN":   15.0,  "DURATION_MAX":   100.0,
    "ADOPTION_MIN":   0.05,  "ADOPTION_MAX":   0.35,
}

API_KEYS = {
    "OPENTOPO":         "7fd208cabed07e635a9dcf8d8c5ca29b",
    "SH_CLIENT_ID":     "sh-e53132df-6521-4fa1-882e-cc4d43f89bc1",
    "SH_CLIENT_SECRET": "YqL56eHEeHVLkbiH8TrLP6eKbIqtUZaB",
}

bbox = (12.14157, 54.08765, 12.15325, 54.09458)

_static = {}

def _worker_init(static_data: dict):
    global _static
    _static = static_data

# --- ACCEPTED CHANGE 1: Multiple Flow Direction (MFD) ---
def compute_upstream_area(dem, res=3.0): # This calculates the distance from the center of each cell to its 8 neighbors, which is essential for determining the slope and flow distribution in the MFD algorithm. The distances are calculated based on the resolution of the grid (res) and the geometry of the neighboring cells. For orthogonal neighbors (up, down, left, right), the distance is simply res. For diagonal neighbors, the distance is res multiplied by sqrt(2) because of the Pythagorean theorem (the diagonal of a square is sqrt(2) times its side length). These distances are used to compute the slope from each cell to its neighbors, which in turn determines how much flow is distributed to each neighbor in the MFD algorithm.
    H, W = dem.shape
    dr   = [-1,-1,-1, 0, 0, 1, 1, 1] # Starting from the center the dr and dc arrays define the relative row and column offsets to access the 8 neighboring cells in a 3x3 grid around a central cell. The order of these offsets corresponds to the following directions: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right. For example, dr[0] = -1 and dc[0] = -1 means that the first neighbor is located one row up and one column to the left of the central cell. This pattern continues for all 8 neighbors, allowing you to easily iterate through them when calculating flow directions and accumulations in the MFD algorithm.
    dc   = [-1, 0, 1,-1, 1,-1, 0, 1]
    dists = np.array([res*np.sqrt(2), res, res*np.sqrt(2),
                      res, res, res*np.sqrt(2), res, res*np.sqrt(2)]) #This calculates the distance from the center of each cell to its 8 neighbors, which is essential for determining the slope and flow distribution in the MFD algorithm. The distances are calculated based on the resolution of the grid (res) and the geometry of the neighboring cells. For orthogonal neighbors (up, down, left, right), the distance is simply res. For diagonal neighbors, the distance is res multiplied by sqrt(2) because of the Pythagorean theorem (the diagonal of a square is sqrt(2) times its side length). These distances are used to compute the slope from each cell to its neighbors, which in turn determines how much flow is distributed to each neighbor in the MFD algorithm.
    
    accum = np.full(H * W, res * res, dtype=np.float64) # np.full's job is to set the initial area to the cell size, which is the minimum contributing area for any cell.H * W: This is the total number of pixels (nodes) in your study area. res * res: This calculates the area of a single pixel (cell) in square meters, assuming 'res' is the length of one side of the pixel in meters. By initializing the accum array with this value, you ensure that each cell starts with at least its own area contributing to it, which is important for accurate flow accumulation calculations.
    #When calculating flooding, every square meter of ground receives rainfall. Before you start "moving" water from high ground to low ground, you must account for the rain that falls directly on each specific spot.The MFD algorithm works by taking a pixel's current area and adding it to the pixels below it. By starting everyone at 9.0 $m^2$, you ensure that a pixel at the bottom of a hill correctly shows the sum of its own rain plus all the rain that flowed down from above.
    # The function np.full is a utility in the NumPy library used to create a new array of a specific shape and type, where every single element is initialized with a "fill value" that you provide.numpy.full(shape, fill_value, dtype=None, order='C'). FOr example: np.full(3, 9.0) = [9, 9, 9] meaning the shape 3 and value of 9 made it.
    order = np.argsort(dem.flatten())[::-1]   # This line flattens the 2D elevation array into a 1D array and then sorts the indices of this array in descending order based on the elevation values. The result is an array of indices that represent the positions of the pixels in the original 2D grid, ordered from highest elevation to lowest. This ordering is crucial for the MFD algorithm, as it ensures that when you iterate through the pixels, you are processing them from the top of the terrain down to the bottom, allowing you to correctly accumulate flow from higher to lower elevations.[::-1] is a slicing operation that reverses the order of the array. In this context, it is used to sort the indices in descending order of elevation, meaning that the algorithm will process the highest points first and then move downwards to the lower points. This is essential for correctly calculating flow accumulation, as water flows from higher elevations to lower elevations.
    # The line above sorts the indices of the flattened elevation array in descending order, which is used in combination with distances and slopes later to determine how water flows from higher to lower elevations in the MFD algorithm. By processing the pixels in this order, you ensure that when you calculate the flow accumulation for a given pixel, you have already calculated the contributions from all pixels that are higher than it, allowing for an accurate representation of how water would accumulate and flow across the terrain.
    for idx in order: # Loop through every pixel in the grid, strictly from highest to lowest. This ensures that when you calculate the flow accumulation for a given pixel, you have already calculated the contributions from all pixels that are higher than it, allowing for an accurate representation of how water would accumulate and flow across the terrain.
        r, c = divmod(idx, W) # The divmod function takes the index of the flattened array and converts it back into 2D row and column indices corresponding to the original elevation grid. This is necessary because the MFD algorithm operates on the 2D grid, and we need to know the specific location of each pixel in terms of its row (r) and column (c) to determine its neighbors and calculate flow directions and accumulations correctly. Here W is the width of the grid, so divmod(idx, W) gives you the row index (r) and column index (c) for the pixel at the flattened index idx.
        slopes = [] #r and c in the line above are the row and column indices of the current pixel being processed. The slopes list is initialized to store the slope values from the current pixel to its valid downhill neighbors. The neighbours list is initialized to store the indices of those valid downhill neighbors. As the loop iterates through each of the 8 neighboring positions around the current pixel, it calculates the slope to each neighbor and checks if it is positive (indicating downhill flow). If it is, the slope value is added to the slopes list, and the index of that neighbor is added to the neighbours list. This information is later used to determine how much flow is distributed to each neighbor based on the relative slopes in the MFD algorithm.
        neighbours = []
        for dri, dci, dist in zip(dr, dc, dists): # This loop iterates through the 8 neighboring positions around the current pixel (r, c) using the relative row and column offsets defined in dr and dc, as well as the corresponding distances in dists. For each neighbor, it calculates the slope from the current pixel to that neighbor by taking the difference in elevation and dividing it by the distance. If the slope is positive (indicating that water would flow downhill), it adds the slope to the slopes list and the index of the neighbor to the neighbours list. This information is later used to determine how much flow is distributed to each neighbor based on the relative slopes. zip here is used to iterate over the dr, dc, and dists arrays simultaneously, allowing you to access the row offset, column offset, and distance for each of the 8 neighboring positions in a single loop. This makes it easier to calculate the slope and determine flow directions for each neighbor in the MFD algorithm.
            rr, cc = r + dri, c + dci
            if 0 <= rr < H and 0 <= cc < W: # Make sure the neighbor is actually inside the map boundaries.
                s = (dem[r, c] - dem[rr, cc]) / dist # dem here is the elevation of the current pixel and the neighbor pixel. By taking the difference in elevation and dividing it by the distance, you get the slope from the current pixel to the neighbor. A positive slope means that water would flow downhill from the current pixel to the neighbor, while a negative slope would indicate uphill flow, which is not possible for water. In the MFD algorithm, only positive slopes are considered for flow distribution, as water can only flow from higher to lower elevations.
                if s > 0:  # Only flow strictly downhill
                    slopes.append(s) # If the slope is positive, it means that water would flow from the current pixel to the neighbor. The slope value is added to the slopes list, which will later be used to determine how much flow is distributed to this neighbor relative to other downhill neighbors. The index of the neighbor (calculated as rr * W + cc) is added to the neighbours list, which keeps track of all valid downhill neighbors for the current pixel. This information is crucial for the MFD algorithm, as it allows you to distribute flow proportionally based on the slopes to each of the valid downhill neighbors.
                    neighbours.append(rr * W + cc) # The index of the neighbor is calculated as rr * W + cc, which converts the 2D row and column indices back into a single index corresponding to the flattened array. This is necessary because the accum array is a 1D array that represents the flow accumulation for each pixel in a flattened format. By storing the indices of the valid downhill neighbors, you can later update their flow accumulation values based on the flow from the current pixel, which is essential for accurately modeling how water flows across the terrain in the MFD algorithm.
        
        if slopes: # If there are valid downhill neighbors, distribute the flow from the current pixel to those neighbors based on the relative slopes. The flow is distributed proportionally to the slope values, meaning that neighbors with steeper slopes will receive more flow. This is done by normalizing the slope values to sum to 1 and then multiplying the current pixel's accumulation by these weights to determine how much flow goes to each neighbor. The accum array is updated for each neighbor accordingly, allowing the flow accumulation to be calculated iteratively as you process each pixel from highest to lowest elevation.
            slopes = np.array(slopes)
            weights = slopes / slopes.sum()  # distribute WATER proportionally to slope
            for nb, w in zip(neighbours, weights): # This loop iterates through each valid downhill neighbor (nb) and its corresponding weight (w) to update the flow accumulation in the accum array. The weight represents the proportion of flow from the current pixel that should be directed to that neighbor based on the relative slope. By multiplying the current pixel's accumulation (accum[idx]) by the weight, you determine how much flow is contributed to that neighbor, and this value is added to accum[nb]. This process allows you to calculate the total flow accumulation for each pixel as you iterate through the grid from highest to lowest elevation, effectively modeling how water would accumulate and flow across the terrain in a realistic manner according to the MFD algorithm.
                accum[nb] += accum[idx] * w # accum[idx] represents the total flow accumulation at the current pixel, which includes the rain that falls directly on it as well as any flow that has accumulated from higher pixels. By multiplying this value by the weight (w), you determine how much of that flow should be directed to the neighbor (nb) based on the relative slope. This value is then added to accum[nb], which updates the flow accumulation for that neighbor pixel. As you process each pixel in descending order of elevation, you effectively model how water flows from higher to lower elevations, allowing you to calculate the total flow accumulation across the entire terrain according to the MFD algorithm.
                
    return accum.reshape(H, W).astype(np.float32) # After processing all pixels, the accum array contains the total flow accumulation for each pixel in a flattened format. By reshaping it back to the original 2D grid shape (H, W) and converting it to float32, you get a 2D array where each value represents the total contributing area (in square meters) for that pixel, which is essential for understanding how water would accumulate and flow across the terrain in the context of flood modeling.

def run_scenario(params: dict) -> str:
    scenario_id    = params["scenario_id"]
    seed           = params["seed"]
    intensity      = params["intensity"]
    duration       = params["duration"]
    adoption_level = params["adoption_level"]

    out_path = os.path.join(OUTPUT_DIR, f"scenario_{scenario_id:05d}.npz")
    if os.path.exists(out_path):
        return f"[{scenario_id:05d}] SKIPPED"

    t0  = time.time()
    rng = np.random.default_rng(seed)

    H               = int(_static["H"])
    W               = int(_static["W"])
    elev_burned     = _static["elev_burned"]
    street_mask     = _static["street_mask"]
    building_mask   = _static["building_mask"]
    water_mask_sink = _static["water_mask_sink"]
    existing_gi     = _static["existing_gi"]
    dist_to_street  = _static["dist_to_street"]
    labeled_b       = _static["labeled_b"]
    num_b           = int(_static["num_b"])
    seg_id_raster   = _static["seg_id_raster"]
    valid_seg_ids   = _static["valid_seg_ids"]
    cfg             = _static["CONFIG"]

    imp   = np.full((H, W), cfg["IMP_URBAN"],  dtype=np.float64)
    n_map = np.full((H, W), cfg["N_DEFAULT"],  dtype=np.float64)
    imp[existing_gi    > 0] = cfg["IMP_EXIST_GI"]
    imp[building_mask  > 0] = cfg["IMP_BUILDING"]
    imp[street_mask    > 0] = cfg["IMP_STREET"]
    imp[water_mask_sink> 0] = 1.0
    n_map[street_mask    > 0] = cfg["N_STREET"]
    n_map[existing_gi    > 0] = cfg["N_GREEN"]
    n_map[water_mask_sink> 0] = cfg["N_WATER"]
    n_map[building_mask  > 0] = cfg["N_BUILDING"]
    gi_map = np.zeros((H, W), dtype=np.int8)

    if num_b > 0:
        TARGET_ROOF_FRACTION = 0.25 
        
        n_roofs   = max(1, int(num_b * adoption_level * TARGET_ROOF_FRACTION))
        chosen    = rng.choice(np.arange(1, num_b + 1),
                               size=n_roofs,
                               replace=False)
        roof_mask = np.isin(labeled_b, chosen)
        imp[roof_mask]    = 0.40
        n_map[roof_mask]  = cfg["N_GREEN"]
        gi_map[roof_mask] = 3

    swale_eligible = (
        (building_mask   == 0) & (street_mask     == 0) &
        (existing_gi     == 0) & (water_mask_sink == 0) &
        (dist_to_street  <= 18.0)
    )
    if np.any(swale_eligible):
        n_swale = max(1, int(swale_eligible.sum() * adoption_level * 0.06 / 5))
        s_rows, s_cols = np.where(swale_eligible)
        starts = rng.choice(len(s_rows), size=min(n_swale, len(s_rows)),
                            replace=False)
        bioswale_mask = np.zeros((H, W), dtype=bool)
        dist_grad_r, dist_grad_c = np.gradient(dist_to_street)
        for idx in starts:
            r, c   = s_rows[idx], s_cols[idx]
            nr, nc = dist_grad_r[r, c], dist_grad_c[r, c]
            norm   = np.hypot(nr, nc)
            if norm < 1e-6:
                continue
            dr = int(np.round(-nc / norm))
            dc = int(np.round( nr / norm))
            if dr == 0 and dc == 0:
                dc = 1
            for step in range(int(rng.integers(4, 11))):
                rr, cc = r + step * dr, c + step * dc
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                if not swale_eligible[rr, cc]:
                    break
                bioswale_mask[rr, cc] = True
        imp[bioswale_mask]    = 0.15
        n_map[bioswale_mask]  = cfg["N_GREEN"]
        gi_map[bioswale_mask] = 1

    if len(valid_seg_ids) > 0:
        n_segs   = len(valid_seg_ids)
        n_chosen = min(max(1, int(n_segs * adoption_level * 0.12)), n_segs)
        chosen_ids = rng.choice(valid_seg_ids, size=n_chosen, replace=False)
        perm_mask  = np.isin(seg_id_raster, chosen_ids) & (street_mask > 0)
        imp[perm_mask]    = 0.25
        n_map[perm_mask]  = cfg["N_GREEN"]
        gi_map[perm_mask] = 2

    mg = RasterModelGrid((H, W), xy_spacing=cfg["RESOLUTION"])
    mg.add_field('topographic__elevation',
                 elev_burned.flatten().astype(np.float64), at='node')
    mg.add_zeros('surface_water__depth', at='node')
    mg.add_field('mannings_n', n_map.flatten(), at='node')
    mg.add_field('mannings_n',
                 mg.map_mean_of_link_nodes_to_link('mannings_n'), at='link')
                 
    # --- ACCEPTED CHANGE 2: Closed Boundaries Except at Streets/Water ---
    # 1. Close all boundaries by default (water bounces back)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,  bottom=mg.BC_NODE_IS_CLOSED)

    # 2. Open up the edges only where there are streets or water bodies
    edge_nodes = np.concatenate([
        mg.nodes_at_top_edge, mg.nodes_at_bottom_edge,
        mg.nodes_at_left_edge, mg.nodes_at_right_edge
    ])
    for node in edge_nodes:
        r, c = divmod(node, W)
        if street_mask[r, c] > 0 or water_mask_sink[r, c] > 0:
            mg.status_at_node[node] = mg.BC_NODE_IS_FIXED_GRADIENT

    # 3. Apply the fixed zero-depth to the sink zone as usual
    water_nodes = np.where(water_mask_sink.flatten() > 0)[0]
    if len(water_nodes) > 0:
        mg.status_at_node[water_nodes] = mg.BC_NODE_IS_FIXED_VALUE
        mg.at_node['surface_water__depth'][water_nodes] = 0.0

    of = OverlandFlow(mg, steep_slopes=True)
    total_t = duration * 60.0
    rain_ms = intensity / (1000.0 * 3600.0)
    INITIAL_ABSTRACTION_MM = 5.0
    cumulative_rain_mm = 0.0
    elapsed = 0.0
    while elapsed < total_t:
            dt = of.calc_time_step()
            if elapsed + dt > total_t:
                dt = total_t - elapsed

            # Apply rain multiplied by the node's specific imperviousness
            active = mg.status_at_node != mg.BC_NODE_IS_FIXED_VALUE
            mg.at_node['surface_water__depth'][active] += (
                rain_ms * dt * imp.flatten()[active])

            of.run_one_step(dt=dt)
            elapsed += dt

    depth = mg.at_node['surface_water__depth'].reshape(H, W).astype(np.float32)
    depth[water_mask_sink > 0] = 0.0

    max_depth  = float(depth.max())
    mean_depth = float(depth[depth >= 0.02].mean()) if np.any(depth >= 0.02) else 0.0

    np.savez_compressed(
        out_path,
        gi_map         = gi_map.astype(np.int8),
        imperviousness = imp.astype(np.float32),
        mannings_n     = n_map.astype(np.float32),
        flood_depth    = depth,
        intensity      = np.float32(intensity),
        duration       = np.float32(duration),
        adoption_level = np.float32(adoption_level),
        scenario_id    = np.int32(scenario_id),
        seed           = np.int32(seed),
        max_depth      = np.float32(max_depth),
        mean_depth     = np.float32(mean_depth),
    )

    return (f"[{scenario_id:05d}] ✅  "
            f"rain={intensity:.1f}mm/h  dur={duration:.0f}min  "
            f"adopt={adoption_level:.2f}  "
            f"max_depth={max_depth:.3f}m  t={time.time()-t0:.1f}s")

if __name__ == '__main__':

    west, south, east, north = bbox
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"ROSTOCK FLOOD BATCH GENERATOR")
    print("=" * 70)

    width_m  = (east - west)  * 111000 * np.cos(np.deg2rad(54.09))
    height_m = (north - south) * 111000
    W = int(width_m  / CONFIG["RESOLUTION"])
    H = int(height_m / CONFIG["RESOLUTION"])
    transform = from_bounds(west, south, east, north, W, H)

    elev = np.ones((H, W), dtype=np.float32) * 12.0
    try:
        url = (f"https://portal.opentopography.org/API/globaldem?demtype=COP30"
               f"&west={west}&south={south}&east={east}&north={north}"
               f"&outputFormat=GTiff&API_Key={API_KEYS['OPENTOPO']}")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open("rostock_elev.tif", "wb") as f:
                f.write(resp.content)
            with rasterio.open("rostock_elev.tif") as src:
                native = src.read(1)
                elev   = zoom(native,
                              (H / native.shape[0], W / native.shape[1]),
                              order=3).astype(np.float32)
                elev   = np.maximum(elev, 0.0)
    except Exception as e:
        print(f"   ⚠️  DEM failed: {e}")

    G_geo = ox.graph_from_bbox(bbox, network_type='drive', simplify=False)
    _, edges_geo = ox.graph_to_gdfs(G_geo)
    street_mask = rasterize(
        [(geom.buffer(0.00006), 1) for geom in edges_geo.geometry],
        out_shape=(H, W), transform=transform, fill=0, dtype=np.float32)

    seg_id_raster = rasterize(
        [(geom.buffer(0.00006), int(i + 1))
         for i, geom in enumerate(edges_geo.geometry)],
        out_shape=(H, W), transform=transform, fill=0, dtype=np.int16)
    valid_seg_ids = np.array(
        [i + 1 for i in range(len(edges_geo))
         if np.any(seg_id_raster == i + 1)], dtype=np.int16)

    building_mask = np.zeros((H, W), dtype=np.float32)
    try:
        building_gdf = ox.features_from_bbox(bbox, tags={'building': True})
        if not building_gdf.empty:
            building_mask = rasterize(
                [(geom, 1) for geom in building_gdf.geometry],
                out_shape=(H, W), transform=transform, fill=0, dtype=np.float32)
    except Exception as e:
        print(f"   ⚠️  Buildings: {e}")

    water_mask = np.zeros((H, W), dtype=np.float32)
    try:
        water_gdf = ox.features_from_bbox(
            bbox, tags={'natural': ['water', 'wetland'],
                        'waterway': ['river', 'canal']})
        if not water_gdf.empty:
            water_gdf['geometry'] = water_gdf.apply(
                lambda x: x.geometry.buffer(0.00005)
                if x.geometry.geom_type in ['LineString', 'MultiLineString']
                else x.geometry, axis=1)
            water_mask = rasterize(
                [(geom, 1) for geom in water_gdf.geometry],
                out_shape=(H, W), transform=transform, fill=0, dtype=np.float32)
    except Exception as e:
        print(f"   ⚠️  Water: {e}")

    quay_mask = np.zeros((H, W), dtype=np.float32)
    try:
        quay_gdf = ox.features_from_bbox(bbox, tags={
            'man_made': ['quay', 'pier', 'breakwater'],
            'landuse':  ['harbour', 'port'],
            'leisure':  ['marina']})
        if not quay_gdf.empty:
            quay_gdf['geometry'] = quay_gdf.apply(
                lambda x: x.geometry.buffer(0.0001)
                if x.geometry.geom_type in ['LineString', 'MultiLineString', 'Point']
                else x.geometry, axis=1)
            quay_mask = rasterize(
                [(geom, 1) for geom in quay_gdf.geometry],
                out_shape=(H, W), transform=transform, fill=0, dtype=np.float32)
    except Exception as e:
        print(f"   ⚠️  Quay: {e}")

    hard_barrier = (building_mask > 0) | (street_mask > 0)
    quay_mask[hard_barrier] = 0
    water_mask_sink = np.maximum(water_mask, quay_mask)
    water_mask_sink[hard_barrier] = 0

    elev_burned = elev.copy()
    elev_burned[water_mask    > 0] -= 2.0
    elev_burned[building_mask > 0] += 5.0
    elev_filled = elev_burned.copy()
    for _ in range(50):
        nb_min = minimum_filter(elev_filled, size=3)
        pit    = elev_filled < nb_min
        if not pit.any():
            break
        elev_filled[pit] = nb_min[pit] + 0.001
    elev_filled[building_mask > 0] = elev_burned[building_mask > 0]
    elev_burned = elev_filled

    dist_to_street = distance_transform_edt(street_mask == 0) * CONFIG["RESOLUTION"]
    labeled_b, num_b = label(building_mask)

    def _get_ndvi():
        try:
            tok = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token",
                data={"grant_type":    "client_credentials",
                      "client_id":     API_KEYS["SH_CLIENT_ID"],
                      "client_secret": API_KEYS["SH_CLIENT_SECRET"]})
            if tok.status_code != 200:
                return None
            token = tok.json()["access_token"]
            evalscript = """//VERSION=3
            function setup(){return{input:["B04","B08"],
            output:{bands:1,sampleType:"FLOAT32"}};}
            function evaluatePixel(s){return[(s.B08-s.B04)/(s.B08+s.B04)];}"""
            payload = {
                "input": {
                    "bounds": {"bbox": list(bbox),
                               "properties": {"crs":
                               "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                    "data": [{"type": "sentinel-2-l2a",
                              "dataFilter": {
                                  "timeRange": {"from": "2023-06-01T00:00:00Z",
                                                "to":   "2023-09-01T00:00:00Z"},
                                  "maxCloudCoverage": 10}}]},
                "output": {"width": W, "height": H,
                           "responses": [{"identifier": "default",
                                          "format": {"type": "image/tiff"}}]},
                "evalscript": evalscript}
            r = requests.post(
                "https://sh.dataspace.copernicus.eu/api/v1/process",
                json=payload,
                headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 200:
                with MemoryFile(r.content) as mf:
                    with mf.open() as src:
                        ndvi = src.read(1)
                        return (ndvi > CONFIG["NDVI_THRESHOLD"]).astype(np.float32)
        except Exception as e:
            print(f"   ⚠️  Sentinel-2: {e}")
        return None

    existing_gi = _get_ndvi()
    if existing_gi is None:
        rng0 = np.random.default_rng(42)
        existing_gi = (
            (rng0.random((H, W)) > 0.80) &
            (street_mask   == 0) &
            (building_mask == 0)
        ).astype(np.float32)

    cols_idx = np.tile(np.arange(W), H)
    rows_idx = np.repeat(np.arange(H), W)
    node_lon = (west  + (cols_idx + 0.5) * (east  - west)  / W).astype(np.float32)
    node_lat = (north - (rows_idx + 0.5) * (north - south) / H).astype(np.float32)

    flat = np.arange(H * W).reshape(H, W)
    r_h  = flat[:, :-1].flatten();  c_h = flat[:, 1:].flatten()
    r_v  = flat[:-1, :].flatten();  c_v = flat[1:,  :].flatten()
    edge_index = np.stack([
        np.concatenate([r_h, c_h, r_v, c_v]),
        np.concatenate([c_h, r_h, c_v, r_v])
    ], axis=0).astype(np.int32)

    upstream_area = compute_upstream_area(elev_burned, res=CONFIG["RESOLUTION"])

    outlet_mask = water_mask_sink > 0
    outlet_mask[0, :] = outlet_mask[-1, :] = True
    outlet_mask[:, 0] = outlet_mask[:, -1] = True
    dist_to_outlet = (distance_transform_edt(~outlet_mask)
                      * CONFIG["RESOLUTION"]).astype(np.float32)

    graph_path = os.path.join(OUTPUT_DIR, "graph_static.npz")
    np.savez_compressed(
        graph_path,
        H=np.int32(H), W=np.int32(W), resolution=np.float32(CONFIG["RESOLUTION"]),
        node_lon=node_lon,      node_lat=node_lat,
        edge_index=edge_index,
        elevation=elev_burned.astype(np.float32),
        street_mask=street_mask.astype(np.float32),
        building_mask=building_mask.astype(np.float32),
        water_mask=water_mask.astype(np.float32),
        water_mask_sink=water_mask_sink.astype(np.float32),
        existing_gi=existing_gi.astype(np.float32),
        dist_to_street=dist_to_street.astype(np.float32),
        dist_to_outlet=dist_to_outlet.astype(np.float32),
        upstream_area=upstream_area.astype(np.float32),
    )

    sampler    = LatinHypercube(d=3, seed=0)
    lhs_scaled = lhs_scale(
        sampler.random(n=N_SCENARIOS),
        [CONFIG["INTENSITY_MIN"], CONFIG["DURATION_MIN"],  CONFIG["ADOPTION_MIN"]],
        [CONFIG["INTENSITY_MAX"], CONFIG["DURATION_MAX"],  CONFIG["ADOPTION_MAX"]]
    )
    scenario_params = [
        {"scenario_id":    i,
         "seed":           i * 1000 + 42,
         "intensity":      float(lhs_scaled[i, 0]),
         "duration":       float(lhs_scaled[i, 1]),
         "adoption_level": float(lhs_scaled[i, 2])}
        for i in range(N_SCENARIOS)
    ]

    static_data = {
        "H": H, "W": W,
        "elev_burned":     elev_burned,
        "street_mask":     street_mask,
        "building_mask":   building_mask,
        "water_mask_sink": water_mask_sink,
        "existing_gi":     existing_gi,
        "dist_to_street":  dist_to_street,
        "labeled_b":       labeled_b,
        "num_b":           num_b,
        "seg_id_raster":   seg_id_raster,
        "valid_seg_ids":   valid_seg_ids,
        "CONFIG":          CONFIG,
    }

    effective_workers = min(N_WORKERS, N_SCENARIOS)
    MIN_FOR_PARALLEL  = max(8, 4 * effective_workers)
    if effective_workers > 1 and N_SCENARIOS < MIN_FOR_PARALLEL:
        effective_workers = 1

    t_start = time.time()

    if effective_workers == 1:
        _worker_init(static_data)
        for p in scenario_params:
            print(f"   {run_scenario(p)}")
    else:
        with ProcessPoolExecutor(
                max_workers=effective_workers,
                initializer=_worker_init,
                initargs=(static_data,)) as executor:
            futures = {executor.submit(run_scenario, p): p["scenario_id"]
                       for p in scenario_params}
            for fut in as_completed(futures):
                try:
                    print(f"   {fut.result()}")
                except Exception as exc:
                    print(f"   [{futures[fut]:05d}] ❌  ERROR: {exc}")

    t_total = time.time() - t_start
    print(f"DONE in {t_total:.1f}s")