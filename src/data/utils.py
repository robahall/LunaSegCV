from pathlib import Path
import csv

from src import ROOT_DIR



def getCandidateInfoList():
    data_dir = Path(ROOT_DIR / 'data')
    mhd_list = data_dir.glob('subset*/*.mhd')
    present_on_disk_set = {p.stem for p in mhd_list}


    with open(data_dir / 'annotations.csv', 'r') as annotation_file:
        diameter_dict = extract_annotation_vals(annotation_file)
    with open(data_dir / 'candidates.csv', 'r') as candidates_file:
        extract_candidates_vals(candidates_file, diameter_dict, present_on_disk_set)

def extract_annotation_vals(file):
    diameter_dict = {}
    for row in list(csv.reader(file))[1:]:
        series_uid = row[0]
        annotation_center_xyz = tuple([float(x) for x in row[1:4]])
        annotation_diameter_mm = float(row[4])
        diameter_dict.setdefault(series_uid, []).append((annotation_center_xyz, annotation_diameter_mm))
    return diameter_dict

def extract_candidates_vals(file, diameter_dict, present_on_disk_set):
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



if __name__ == "__main__":
    getCandidateInfoList()












