import csv
import functools
from pathlib import Path
from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

from src import ROOT_DIR


CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule, diameter_mm, series_uid, center_xyz'
                                )
IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


@functools.lru_cache(1)
def getCandidateInfoList():
    data_dir = Path(ROOT_DIR / 'data')
    mhd_list = data_dir.glob('subset*/*.mhd')
    present_on_disk_set = {p.stem for p in mhd_list}


    with open(data_dir / 'annotations.csv', 'r') as annotation_file:
        diameter_dict = extract_annotation_vals(annotation_file)
    with open(data_dir / 'candidates.csv', 'r') as candidates_file:
        candidate_info_list = extract_candidates_vals(candidates_file, diameter_dict, present_on_disk_set)

    return candidate_info_list

def extract_annotation_vals(file):
    diameter_dict = {}
    for row in list(csv.reader(file))[1:]:
        series_uid = row[0]
        annotation_center_xyz = tuple([float(x) for x in row[1:4]])
        annotation_diameter_mm = float(row[4])
        diameter_dict.setdefault(series_uid, []).append((annotation_center_xyz, annotation_diameter_mm))
    return diameter_dict

def extract_candidates_vals(file, diameter_dict, present_on_disk_set):
    candidate_info_list = []
    for row in list(csv.reader(file))[1:]:
        series_uid = row[0]
        if series_uid not in present_on_disk_set:
            continue
        candidate_center_xyz = tuple([float(x) for x in row[1:4]])
        is_nodule = bool(int(row[4]))
        candidate_diameter_mm = 0.0

        for annotation_tup in diameter_dict.get(series_uid, []):
            annotation_center_xyz, annotation_diameter_mm = annotation_tup
            for i in range(len(annotation_center_xyz)):
                delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                if delta_mm > annotation_diameter_mm/4:
                    break
                else:
                    candidate_diameter_mm = annotation_diameter_mm
                    break
        candidate_info_list.append(CandidateInfoTuple(
            is_nodule, candidate_diameter_mm, series_uid, candidate_center_xyz
            ))
    candidate_info_list.sort(reverse=True)
    return candidate_info_list

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

class Ct:

    def __init__(self, series_uid):
        mhd_path = Path(ROOT_DIR / 'data/luna').glob(f'subset*/{series_uid}.mhd')
        ct_mhd = sitk.ReadImage(str(next(mhd_path)))
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self, ratio_int=0, candidateInfo_list=None,):
        self.ratio_int = ratio_int
        self.candidateInfo_list = candidateInfo_list
        self.negative_list = [nt for nt in self.candidateInfo_list if not nt.isNodule]
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule]

    def __len__(self):
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int +1)
            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.negative_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        width_irc = (32, 48, 48)
        ct = getCt(candidateInfo_tup.series_uid)
        candidate_a, center_irc = ct.getRawCandidate( candidateInfo_tup.center_xyz, width_irc,)
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule,
                candidateInfo_tup.isNodule
            ],
            dtype=torch.long,
        )

        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)

## Next set up validation set.

if __name__ == "__main__":
    ds = LunaDataset(candidateInfo_list=getCandidateInfoList())
    print(len(ds))










