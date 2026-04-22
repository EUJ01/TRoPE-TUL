import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pyproj import Proj, transform

TRAJ_ID_COL = 'traj_id'
USER_COL = 'user_id'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'

ST_MAP = {
    "spatial": [0, 1],
    "temporal": [2, 3]
}

FEATURE_TOKEN = 0
FEATURE_PAD = 0
PAD_TOKEN = 5

def coord_transform_GPS_UTM(traj, UTM_region, origin_coord="latlong", dest_coord="utm"):
    """Transforms GPS coordinates to UTM or vice-versa."""
    if origin_coord == "latlong":
        origin = Proj(proj="latlong", datum="WGS84")
        dest = Proj(proj="utm", zone=UTM_region, datum="WGS84")
    elif origin_coord == "utm":
        dest = Proj(proj="latlong", datum="WGS84")
        origin = Proj(proj="utm", zone=UTM_region, datum="WGS84")
    else:
        raise NotImplementedError(f"Coordinate type '{origin_coord}' not supported.")

    # Apply transformation based on array dimensions
    if traj.ndim == 2:
        easting, northing = transform(origin, dest, traj[:, 0], traj[:, 1]) #type: ignore
        traj[:, 0], traj[:, 1] = easting, northing
    elif traj.ndim == 3:
        easting, northing = transform(origin, dest, traj[:, :, 0], traj[:, :, 1]) #type: ignore
        traj[:, :, 0], traj[:, :, 1] = easting, northing
        
    return traj

class TRoPETULDataset(Dataset):
    def __init__(self, traj_df, UTM_region, scale, spatial_middle_coord=None):
        super().__init__()
        self.traj_df = traj_df
        self.UTM_region = UTM_region
        self.scale = scale
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

        spatial_border = traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]
        
        # Calculate or assign the middle coordinate
        if spatial_middle_coord is None:
            self.middle_coord = np.array([[(self.spatial_border[0][0] + self.spatial_border[1][0]) / 2, 
                                           (self.spatial_border[0][1] + self.spatial_border[1][1]) / 2]])
            self.spatial_middle_coord = coord_transform_GPS_UTM(self.middle_coord, self.UTM_region)
        else:
            self.spatial_middle_coord = spatial_middle_coord

        # Transform and scale coordinates
        traj_gps = traj_df[[X_COL, Y_COL]].values.copy()
        traj_utm = (coord_transform_GPS_UTM(traj_gps, self.UTM_region) - self.spatial_middle_coord) / self.scale 
        self.traj_df[[X_COL, Y_COL]] = pd.DataFrame(traj_utm)

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        one_traj = self.traj_df[self.traj_df[TRAJ_ID_COL] == self.traj_ids[index]].copy()
        one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
        return one_traj

class TULPadder:
    """Collate function for Trajectory User Classification task.""" 
    _current_row_indices = None  # Explicit declaration of the class variable

    def __init__(self, num_users):
        self.num_users = num_users
    
    def __call__(self, raw_batch):
        input_batch, output_batch, pos_batch, row_indices_batch = [], [], [], []
        
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()
            user_id = traj[USER_COL].iloc[0] 
            L_out = len(traj)
            
            # Extract original CSV row indices for POI embedding lookup
            row_indices_batch.append(traj.index.values)
            
            # Build input sequence
            input_row = np.stack([traj_feats, np.ones_like(traj_feats) * FEATURE_TOKEN], axis=-1)
            input_batch.append(input_row)
            
            # Build output one-hot target
            output_row = np.zeros((1, self.num_users))  
            output_row[0, user_id] = 1  
            output_batch.append(output_row)
            
            # Build position sequences
            pos_batch.append(np.array([[i, 0] for i in range(L_out)]))

        # Convert lists to padded tensors
        input_batch = torch.tensor(pad_batch_3d(input_batch), dtype=torch.float32) 
        output_batch = torch.tensor(np.array(output_batch), dtype=torch.float32).squeeze(1) 
        pos_batch = torch.tensor(pad_batch_2d(pos_batch), dtype=torch.long)
        
        # Pad row_indices to match sequence length (B, L_max)
        max_len = input_batch.size(1)
        padded_row_indices = np.full((len(row_indices_batch), max_len), -1, dtype=np.int64)
        for i, ri in enumerate(row_indices_batch):
            padded_row_indices[i, :len(ri)] = ri
            
        # WARNING: This requires dataloader `num_workers=0` to work properly!
        TULPadder._current_row_indices = torch.tensor(padded_row_indices, dtype=torch.long)
        
        return input_batch, output_batch, pos_batch

def fetch_task_padder(padder_name, padder_params):
    if padder_name == 'tul':
        return TULPadder(**padder_params)
    raise NotImplementedError(f"No Padder named '{padder_name}'")

def pad_batch_3d(batch):
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len, batch[0].shape[1]), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len, batch[0].shape[1]), PAD_TOKEN, dtype=float)
    ), axis=-1)

    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr
    return padded_batch

def pad_batch_2d(batch):
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    ), axis=-1)

    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr
    return padded_batch