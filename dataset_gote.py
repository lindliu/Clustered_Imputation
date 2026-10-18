import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
from utils import get_randmask, get_hist_mask

import matplotlib.pyplot as plt

def get_mean_std():
    # Fit on train + validation days (the first 80%); never use test days.
    df = pd.read_csv("./data/goteborg/gote_ground.csv", index_col="m_time", parse_dates=True)
    days = np.sort(df.index.normalize().unique())
    trainval_days = days[:int(len(days) * 0.8)]
    trainval_df = df[df.index.normalize().isin(trainval_days)]
    mean = trainval_df.mean().to_numpy()
    std = trainval_df.std(ddof=0).to_numpy()  # Standard deviation, not variance.
    std[std == 0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("A sensor has no valid train/validation observations.")
    with open("./data/goteborg/gote_meanstd.pk", "wb") as f:
        pickle.dump((mean, std), f)
        
class Gote_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=53, mode="train", val_len=0.1, is_interpolate=False,
                 target_strategy='hybrid', mask_sensor=None, missing_ratio=None):
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.mask_sensor = mask_sensor

        path = "./data/goteborg/gote_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
            
        df = pd.read_csv(
            "./data/goteborg/gote_ground.csv",
            index_col="m_time",
            parse_dates=True,
        )
        df_gt = pd.read_csv(
            "./data/goteborg/gote_missing.csv",
            index_col="m_time",
            parse_dates=True,
        )

        df, day_table, code_map = add_compact_day_code(df)
        df_gt, _, _ = add_compact_day_code(df_gt)
        #df, ym_table = add_compact_ym_code(df)
        #df_gt, _ = add_compact_ym_code(df_gt)
        month_list_all = np.sort(df['day_code'].unique())
        
        #print('data : ',len(month_list_all))
        if not 0 < val_len < 0.8:
            raise ValueError("val_len must be between 0 and 0.8.")
        num_train = int(len(month_list_all) * (0.8 - val_len))
        test_start = int(len(month_list_all) * 0.8)
        if mode == "train":
            month_list = month_list_all[:num_train]
            # 1st,4th,7th,10th months are excluded from histmask (since the months are used for creating missing patterns in test dataset)
            flag_for_histmask = np.random.choice(2,num_train) #[0, 1, 0, 1]
        elif mode == "valid":
            month_list = month_list_all[num_train:test_start]
        elif mode == "test":
            month_list = month_list_all[test_start:]
            # month_list = [5]
        self.month_list = month_list

        # create data for batch
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets
        
        self.timepoints = []
        

        for i in range(len(month_list)):
            current_df = df[df['day_code'] == month_list[i]]
            current_df_gt = df_gt[df_gt['day_code'] == month_list[i]]
            
            current_df = current_df.drop(columns=["day_code"])
            current_df_gt = current_df_gt.drop(columns=["day_code"])
            
            # Whole days belong to one split; do not take a second validation tail.
            if len(current_df) < eval_length:
                raise ValueError(f"Day {month_list[i]} is shorter than eval_length={eval_length}.")
            current_length = len(current_df) - eval_length + 1

            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            if len(self.mask_sensor) > 0:
                for sensor in self.mask_sensor:
                    c_gt_mask[:, sensor] = 0
                if self.mode == 'train':
                    for sensor in self.mask_sensor:
                        c_mask[:, sensor] = 0
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask
            #c_data = (
            #    current_df.fillna(0).values
            #) * c_mask
            #print(c_data.shape)
            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)
            self.timepoints.append((current_df.index.hour))
            
            if mode == "test":
                n_sample = len(current_df) // eval_length
                # interval size is eval_length (missing values are imputed only once)
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        # masks for 1st,4th,7th,10th months are used for creating missing patterns in test data,
        # so these months are excluded from histmask to avoid leakage
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:  # dummy (histmask is only used for training)
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month

        
    def __getitem__(self, org_index):
        #print(self.index_month)
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]

        index2 = np.random.randint(0, len(self.use_index))
        hist_month = self.index_month_histmask[index2]
        hist_index = self.position_in_month_histmask[index2]

        ob_data = self.observed_data[c_month][c_index:c_index + self.eval_length]
        ob_mask = self.observed_mask[c_month][c_index:c_index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[c_month][c_index:c_index + self.eval_length]
        for_pattern_mask = self.observed_mask[hist_month][hist_index:hist_index + self.eval_length]
        #print(ob_data.shape, ob_mask.shape, ob_mask_t.shape, gt_mask.shape)
        timepoints = np.array(self.timepoints[c_month][c_index:c_index + self.eval_length])

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_hist_mask(ob_mask_t, for_pattern_mask=for_pattern_mask)
            else:
                cond_mask = get_randmask(ob_mask_t)
        
        
        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "hist_mask": for_pattern_mask,
            "timepoints": np.arange(self.eval_length),#timepoints,#np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask.numpy()
        }
        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            #print(cond_mask.shape, ob_data.shape)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()
        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(batch_size, device, val_len=0.1, is_interpolate=False, num_workers=4, target_strategy='hybrid', mask_sensor=None):
    dataset = Gote_Dataset(mode="train", val_len=val_len, is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = Gote_Dataset(mode="test", val_len=val_len, is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = Gote_Dataset(mode="valid", val_len=val_len, is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler

  

def add_compact_day_code(df: pd.DataFrame, col_name="day_code"):
    day = pd.to_datetime(df.index).normalize()  # day bucket

    # factorize gives codes 0..K-1 and the uniques (in first-seen order)
    codes, uniques = pd.factorize(day, sort=True)  # sort=True => chronological

    df = df.copy()
    df[col_name] = codes.astype(np.int64)

    day_table = pd.DataFrame({
        col_name: np.arange(len(uniques), dtype=np.int64),
        "date": pd.to_datetime(uniques),  # ensure Timestamp dtype
    })
    day_table["date_str"] = day_table["date"].dt.strftime("%Y-%m-%d")
    day_table["year"] = day_table["date"].dt.year
    day_table["month"] = day_table["date"].dt.month
    day_table["day"] = day_table["date"].dt.day

    # Optional: build a mapping back from date->code if you need it
    code_map = dict(zip(day_table["date"].tolist(), day_table[col_name].tolist()))

    return df, day_table, code_map
    
    
if __name__ == '__main__':

	dataset, train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
	    1, device='cpu', val_len=36,
	    is_interpolate=False, num_workers=4,
	    target_strategy='hybrid', mask_sensor=[]
	)
	#print(len(train_loader))
	#print(dataset[0]['observed_data'].shape)
        
	#print(dataset.index_month)
	#print((np.array(dataset.index_month)==0).sum())
	#print((np.array(dataset.index_month)==1).sum())
	#print((np.array(dataset.index_month)==2).sum())
	#print((np.array(dataset.index_month)==3).sum())
	
	df = pd.read_csv(
            "./data/goteborg/gote_ground.csv",
            index_col="m_time",
            parse_dates=True,
        )
	df, day_table, code_map = add_compact_day_code(df)

	#print((df['ym_code']==0).sum())
	#print((df['ym_code']==1).sum())
	#print((df['ym_code']==2).sum())
	#print((df['ym_code']==3).sum())		
	
	plt.plot(dataset[0]['timepoints'],dataset[399]['observed_data'][:,0], 'o')
	plt.plot(np.arange(36), df[df['day_code']==4]['Cam1'][0:36+0], 'o')
	plt.savefig('test.png')

    
    
    
    
    
    
    
    
    
    
    
    
