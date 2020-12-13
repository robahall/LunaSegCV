## Utilizing CV/Segin PyTorch Lightning

Data Files:

* candidates.csv - All information about all the lumps in CT scan whether they are malignant, benign or something else. 
    * seriesuid - CT Scan ID (str)
    * coordX - X coordinate (float)
    * coordY - Y coordinate (float)
    * coordZ - Z coordinate (float)
    * class - Malignant or benign (bool). 
* annoatations.csv - All information about some of the candidates that have been flagged as nodules. 
    * seriesuid - CT Scan ID (str)
    * coordX - X coordinate (float)
    * coordY - Y coordinate (float)
    * coordZ - Z coordinate (float)
    * diameter_mm - size of nodules (float)
* .mhd - 

* .raw - 


