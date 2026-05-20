# Import packages for use:
import numpy as np
import pandas as pd
import ngl_tools.smt as smt

pd.set_option('display.max_columns', None)


def gamma_t (depth, fs, qt):

    # Replace invalid values in CPT_fs and CPT_qt with NaN
    fs1 = fs.copy()
    qt1 = qt.copy()
    fs1[fs1 <= 0] = np.nan
    qt1[qt1 <= 0] = np.nan

    # Compute gamma only for valid values
    valid_mask = ~np.isnan(fs1) & ~np.isnan(qt1)

    gamma = np.full_like(depth, np.nan)  # Initialize with NaN

    # Calculate gamma using the formula
    # gamma[valid_mask] = (11.46 + 0.33 * np.log10(depth[valid_mask]) + 3.1 * np.log10(fs1[valid_mask]) + 0.7 * np.log10(qt1[valid_mask]))
    gamma[valid_mask] = 9.81 * (0.27 * np.log10(fs1[valid_mask]/qt1[valid_mask] * 100) + 0.36 * np.log10(qt1[valid_mask] / 101.3) + 1.236)

     # Interpolate NaN values
    gamma = np.interp(depth, depth[valid_mask], gamma[valid_mask])

    return gamma


def get_CPT (CPT_depth, CPT_qt, CPT_fs, dGWT):
    # TT_Meta, TT_Data, CPT_meta, CPT_data, TT_depth, TT, CPT_ID, CPT_depth, CPT_qt, CPT_fs = query_TT(travelTimeMeta_ID)

    # dGWT = query_DGWT(travelTimeMeta_ID)
    CPT_fs = CPT_fs * 1000 #Convert to kPa
    CPT_qt = CPT_qt * 1000 #Convert to kPa

    pa = 101.3 #KPa,
    gamma = gamma_t(CPT_depth, CPT_fs, CPT_qt)

    CPT_fs[CPT_fs < 0] = 0.01
    CPT_qt[CPT_qt < 0] = 0.01

    # integrate gamma over CPT_depth to get sigmav, then compute sigmavp by subtracting the pore pressure u
    # updated 2026-02-24
    depth_interval = np.diff(CPT_depth, prepend=0)
    sigmav = np.cumsum(gamma * depth_interval)
           
    u = (CPT_depth - dGWT) * 9.81
    u[u<0] = 0
    sigmavp = sigmav - u

    # Compute fz
    fz = sigmavp/pa

    # Generate the layered Ic
    Ic, Qtn, Fr = smt.get_Ic_Qtn_Fr(CPT_qt, CPT_fs, sigmav, sigmavp)
    qt_inv, fs_inv, Ic_inv = smt.cpt_inverse_filter(CPT_qt, CPT_depth, fs=CPT_fs, sigmav=sigmav, sigmavp=sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = smt.get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)
    ztop, zbot, Qtn_lay, Ic_lay = smt.cpt_layering(Qtn_inv, Ic_inv, CPT_depth, dGWT=dGWT, Nmin = 1, Nmax = None, averaging = 1)

    # Resample Ic_lay to CPT_depth
    Ic_lay_resampled = np.zeros(len(CPT_depth))
    Qtn_lay_resampled = np.zeros(len(CPT_depth))
    for i in range(len(ztop)):
        for j in range(len(CPT_depth)):
            if CPT_depth[j] >= ztop[i] and CPT_depth[j] < zbot[i]:
                Qtn_lay_resampled[j] = Qtn_lay[i]
                Ic_lay_resampled[j] = Ic_lay[i]
    Ic_lay_resampled[-1] = Ic_lay[-1]
    Qtn_lay_resampled[-1] = Qtn_lay[-1]


    Qtncs = Qtn_lay_resampled.copy()
    kc = np.full(Qtn_lay_resampled.shape, 1.0)
    kc = 1 + 5 / (1 + np.exp( - 14 * (Ic_lay_resampled - 2.1) ))
    Qtncs = Qtn_lay_resampled * kc

    return fz, Ic_lay_resampled, Qtncs, CPT_depth, Qtn_inv, Ic_inv, Qtn, Ic