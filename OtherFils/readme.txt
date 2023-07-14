- data (./data/)
    1. AD
       - 87ADs_before_filter1000.mat: idx 0:86
       - DK_timecourse.mat: 87
       - ADFiltered.pkl: 0-87, data after filtering and truncate the first 2000 pts.
    2. Ctrl
        - 70Ctrls_before_filter1000.mat: 0:69
        - timecourse_ucsfCONT_group.mat: 70:91
            Including the RIDs list for the correcting order with the data
       - CtrlFiltered.pkl: 0-91, data after filtering and truncate the first 2000 pts.

- Order of fils
    1. radid_AD.mat: include the radid of the AD group  with the same order
        i.e., 87ADs_before_filter1000.mat and DK_timecourse.mat  
    2. radid_control.mat: include the radid of the Ctrl group  with the same order
        i.e., 70Ctrls_before_filter1000.mat
    2. radid_control_add.mat: include the radid of the Ctrl group  with the same order
        i.e., timecourse_ucsfCONT_group.mat

- Baseline info
    1. 88 vs 70 (previous)
        - ADcomplete.csv and Ctrlcomplete.csv
    2. All 
        - AllDataSelBaseline.csv: RID, AGE, gender for all the data used in our analysis
            Note that we include all AD datasets, but select part of the control datasets.
            it is generated in `Select Data.ipynb`
        - AllDataSelBaselineOrdered.csv: All baseline info for 88+92 datasets.
            It is ordered as: 87ADs_before_filter1000.mat ->DK_timecourse.mat -> 70Ctrls_before_filter1000.mat -> - timecourse_ucsfCONT_group.mat
            It is generated in `Select Data.ipynb`
        - AllDataSelBaselineOrdered_r_ncpt.csv: in results folder.
            AllDataSelBaselineOrdered_r_ncpt.csv + rank + ncpts
            It is generated in `Select Data.ipynb`
            

- KpIdxsAll.pkl: selected index from AD and Ctrl, all datasets
