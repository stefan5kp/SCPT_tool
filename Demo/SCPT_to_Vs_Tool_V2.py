# Import packages for use:
import pwlf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ngl_tools.smt as smt
from itertools import combinations

pd.set_option('display.max_columns', None)

VSPDB_TABLE_DIR = r"C:/Users/5_kp/OneDrive - UCLA IT Services/PhD in Civil Engineering UCLA/04 First Year 24-25/01 VSPDB/Source Files/VSPDB 1.0 Tables/"

SITE     = pd.read_csv(VSPDB_TABLE_DIR + 'site.csv').rename(columns={'name':'site_name','latitude':'site_latitude','longitude':'site_longitude'})
CITATION = pd.read_csv(VSPDB_TABLE_DIR + 'citation.csv')
CPT_META = pd.read_csv(VSPDB_TABLE_DIR + 'conePenetrationTestMeta.csv')
CPT_DATA = pd.read_csv(VSPDB_TABLE_DIR + 'conePenetrationTestArray.csv')
TT_META = pd.read_csv(VSPDB_TABLE_DIR + 'travelTimeMeta.csv')
TT_DATA = pd.read_csv(VSPDB_TABLE_DIR + 'travelTimeArray.csv')
DGWT_DATA = pd.read_csv(VSPDB_TABLE_DIR + 'DGWT.csv')
VEL_META = pd.read_csv(VSPDB_TABLE_DIR + 'velocityProfileMeta.csv')
VEL_DATA = pd.read_csv(VSPDB_TABLE_DIR + 'velocityProfileArray.csv')

try:
    REVIEW_DF = pd.read_csv("SCPT_Processing.csv")
except:
    # It is verified that the travelTimeMeta_ID and site_ID is one-to-one relation
    REVIEW_DF = TT_META.copy()[['travelTimeMeta_ID','site_ID']] 
    REVIEW_DF.insert(2, 'status', ['Pending']*len(REVIEW_DF), True)
    REVIEW_DF.insert(3, 'c0', [np.nan]*len(REVIEW_DF), True)
    REVIEW_DF.insert(4, 'c1', [np.nan]*len(REVIEW_DF), True)
    REVIEW_DF.insert(5, 'c2', [np.nan]*len(REVIEW_DF), True)
    REVIEW_DF.insert(6, 'Misfit', [np.nan]*len(REVIEW_DF), True)
    REVIEW_DF.insert(7, 'comment', [np.nan]*len(REVIEW_DF), True)
    REVIEW_DF.to_csv("SCPT_Processing_V2.csv", index=False, header=True)

try: 
    VS_CPT_DF = pd.read_csv("SCPT_Vs.csv")
except:
    VS_CPT_DF = pd.DataFrame(columns=['travelTimeMeta_ID', 'Vs_lay', 'CPT_depth'])
    VS_CPT_DF.to_csv("SCPT_Vs.csv", index=False, header=True)

try:
    VS_SLOPEBREAK_DF = pd.read_csv("SLOPEBREAK_Vs.csv")
except:
    VS_SLOPEBREAK_DF = pd.DataFrame(columns=['travelTimeMeta_ID', 'Velocity', 'Top_depth', 'Bottom_depth'])
    VS_SLOPEBREAK_DF.to_csv("SLOPEBREAK_Vs.csv", index=False, header=True)

META = REVIEW_DF[['travelTimeMeta_ID','site_ID']]
META = META.merge(CPT_META, on=['site_ID'], how='left')
META = META.merge(CITATION, on=['citation_ID'], how='left')
META.to_csv("META.csv", index=False, header=True)

def query_CPT(conePenetrationTestMeta_ID):
    id = conePenetrationTestMeta_ID
    meta = CPT_META[CPT_META['conePenetrationTestMeta_ID'] == id].reset_index(drop=True)
    data = CPT_DATA[CPT_DATA['conePenetrationTestMeta_ID'] == id].reset_index(drop=True)
    Depth = data['depth'].values
    qt = data['tipResistance'].values
    fs = data['sleeveFriction'].values
    # take out when any of Depth, qt, fs is NaN
    valid_mask = ~np.isnan(Depth) & ~np.isnan(qt) & ~np.isnan(fs)
    Depth = Depth[valid_mask]
    fs = fs[valid_mask]

    return(meta, data, Depth, qt, fs)

def query_TT(travelTimeMeta_ID):
    id = travelTimeMeta_ID
    tt_meta = TT_META[TT_META['travelTimeMeta_ID'] == id].reset_index(drop=True)
    tt_data = TT_DATA[TT_DATA['travelTimeMeta_ID'] == id].reset_index(drop=True)
    tt_depth = tt_data['depth'].values
    traveltime = tt_data['traveltime'].values
    # take out when any of tt_depth, traveltime is NaN
    valid_mask = ~np.isnan(tt_depth) & ~np.isnan(traveltime)
    tt_depth = tt_depth[valid_mask]
    traveltime = traveltime[valid_mask]

    cpt_id = META[META['travelTimeMeta_ID'] == id]['conePenetrationTestMeta_ID'].astype('float64').iloc[0]
    cpt_meta1, cpt_data1, cpt_depth1, qt1, fs1 = query_CPT(cpt_id)

    return(tt_meta, tt_data, cpt_meta1, cpt_data1, tt_depth, traveltime, cpt_id, cpt_depth1, qt1, fs1)


def query_Vel(velocityProfileMeta_ID):
    id = velocityProfileMeta_ID
    vel_meta = VEL_META[VEL_META['velocityProfileMeta_ID'] == id].reset_index(drop=True)
    vel_data = VEL_DATA[VEL_DATA['velocityProfileMeta_ID'] == id].reset_index(drop=True)
    vel_unit = vel_meta['depthTop_unit'].values
    vel_depth = vel_data['depthTop'].values
    suspension_logging = vel_data['value'].values

    return(vel_depth, suspension_logging, vel_unit)

def query_DGWT(travelTimeMeta_ID):
    id = travelTimeMeta_ID
    DGWT_data = DGWT_DATA[DGWT_DATA['travelTimeMeta_ID'] == id].reset_index(drop=True)
    DGWT = DGWT_data['DGWT'].iloc[0]
    
    return DGWT

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

def interleave(v1, v2):
    vout = np.empty(2 * len(v1), dtype=float)
    vout[0::2] = v1
    vout[1::2] = v2
    return vout 

# Function to get sigmavp, Qtn and Ic

def Preprocessing_CPT (travelTimeMeta_ID):
    TT_Meta, TT_Data, CPT_meta, CPT_data, TT_depth, TT, CPT_ID, CPT_depth, CPT_qt, CPT_fs = query_TT(travelTimeMeta_ID)

    dGWT = query_DGWT(travelTimeMeta_ID)
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
    qt_inv, fs_inv, Ic_inv = smt.cpt_inverse_filter(CPT_qt, CPT_depth, fs=CPT_fs, sigmav=sigmav, sigmavp=sigmavp)
    Ic, Qtn, Fr = smt.get_Ic_Qtn_Fr(CPT_qt, CPT_fs, sigmav, sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = smt.get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)
    FC = smt.get_FC_from_Ic(Ic_inv, 0.0)
    qc1N, qc1Ncs = smt.get_qc1N_qc1Ncs(CPT_qt, CPT_fs, sigmav, sigmavp, FC)
    qc1N_inv, qc1Ncs_inv = smt.get_qc1N_qc1Ncs(qt_inv, fs_inv, sigmav, sigmavp, FC)
    # ztop, zbot, qc1Ncs_lay, Ic_lay = smt.cpt_layering(qc1Ncs_inv, Ic_inv, CPT_depth, dGWT=dGWT, Nmin = 1, Nmax = None, averaging = 1)
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

    return fz, Qtn_lay_resampled, Ic_lay_resampled, Qtn_lay, Ic_lay, Qtn_inv, Ic_inv, CPT_qt, CPT_depth, ztop, zbot

def Preprocessing_TT (travelTimeMeta_ID, source = 1.0):
    TT_Meta, TT_Data, CPT_meta, CPT_data, TT_depth, Meas_TT, CPT_ID, CPT_depth, CPT_qt, CPT_fs = query_TT(travelTimeMeta_ID)

    # Correction of Source Offset Distance
    TT = TT_depth / np.sqrt (TT_depth**2 + source**2) * Meas_TT

    slowness_TT = np.zeros(len(TT)-1)
    slowness_TT_depth = np.zeros(len(TT)-1)
    for i in range(len(TT)-1):
        slowness_TT[i] = (TT[i+1] - TT[i]) / (TT_depth[i+1] - TT_depth[i]) /1000
        slowness_TT_depth[i] = (TT_depth[i+1] + TT_depth[i]) / 2
  
    ztop_TT = TT_depth[:-1]
    zbot_TT = TT_depth[1:]

    halfdepth = slowness_TT_depth

    DTT = np.diff(TT)

    return TT, TT_depth, halfdepth, DTT, slowness_TT, slowness_TT_depth, ztop_TT, zbot_TT

def Vs_interval (traveltimeMeta_ID):
    TT, TT_depth, halfdepth, DTT, slowness_TT, slowness_TT_depth, ztop_TT, zbot_TT = Preprocessing_TT(traveltimeMeta_ID)
    Vsi = (zbot_TT - ztop_TT)/(DTT/1000)
    return Vsi

def Robertson (Qtn, Ic, fz):
    Vs = np.exp(1.93 + 0.5 * np.log(Qtn) + 0.25 * np.log(fz) + 0.63 * Ic)
    return Vs

def get_CPT (travelTimeMeta_ID):
    TT_Meta, TT_Data, CPT_meta, CPT_data, TT_depth, TT, CPT_ID, CPT_depth, CPT_qt, CPT_fs = query_TT(travelTimeMeta_ID)

    dGWT = query_DGWT(travelTimeMeta_ID)
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
    qt_inv, fs_inv, Ic_inv = smt.cpt_inverse_filter(CPT_qt, CPT_depth, fs=CPT_fs, sigmav=sigmav, sigmavp=sigmavp)
    Ic, Qtn, Fr = smt.get_Ic_Qtn_Fr(CPT_qt, CPT_fs, sigmav, sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = smt.get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)
    FC = smt.get_FC_from_Ic(Ic_inv, 0.0)
    qc1N, qc1Ncs = smt.get_qc1N_qc1Ncs(CPT_qt, CPT_fs, sigmav, sigmavp, FC)
    qc1N_inv, qc1Ncs_inv = smt.get_qc1N_qc1Ncs(qt_inv, fs_inv, sigmav, sigmavp, FC)
    # ztop, zbot, qc1Ncs_lay, Ic_lay = smt.cpt_layering(qc1Ncs_inv, Ic_inv, CPT_depth, dGWT=dGWT, Nmin = 1, Nmax = None, averaging = 1)
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

    # Clean sand correction (Robertson 2021)
    # index1= (Ic_lay_resampled > 1.7) & (Ic_lay_resampled < 3.0)
    # index2 = (Ic_lay_resampled >= 3.0)
    Qtncs = Qtn_lay_resampled.copy()
    kc = np.full(Qtn_lay_resampled.shape, 1.0)

    # kc[index1] = 1.8346 * Ic_lay_resampled[index1] ** 5 - 23.673 * Ic_lay_resampled[index1] ** 4 + 124.02 * Ic_lay_resampled[index1] ** 3 - 320.616 * Ic_lay_resampled[index1] ** 2 + 405.821 * Ic_lay_resampled[index1] - 199.97
    # kc[index2] = 1.8346 * 3.0 ** 5 - 23.673 * 3.0 ** 4 + 124.02 * 3.0 ** 3 - 320.616 * 3.0 ** 2 + 405.821 * 3.0 - 199.97

    kc = 1 + 5 / (1 + np.exp( - 14 * (Ic_lay_resampled - 2.1) ))
    # kc = 1 + 4 / (1 + np.exp( - 6 * (Ic_lay_resampled - 2.5) ))
    # kc = 1 + 1 / (1 + np.exp( - 4 * (Ic_lay_resampled - 2.35) ))
    # kc = 1 + 2 / (1 + np.exp( - 4.5 * (Ic_lay_resampled - 2.4) ))

    Qtncs = Qtn_lay_resampled * kc

    return fz, Ic_lay_resampled, Qtncs, CPT_depth, Qtn_inv, Ic_inv

def get_CPT_KC (travelTimeMeta_ID, Kc_c1, Kc_c2): # for testing the clean sand correction
    TT_Meta, TT_Data, CPT_meta, CPT_data, TT_depth, TT, CPT_ID, CPT_depth, CPT_qt, CPT_fs = query_TT(travelTimeMeta_ID)

    dGWT = query_DGWT(travelTimeMeta_ID)
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
    qt_inv, fs_inv, Ic_inv = smt.cpt_inverse_filter(CPT_qt, CPT_depth, fs=CPT_fs, sigmav=sigmav, sigmavp=sigmavp)
    Ic, Qtn, Fr = smt.get_Ic_Qtn_Fr(CPT_qt, CPT_fs, sigmav, sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = smt.get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)
    FC = smt.get_FC_from_Ic(Ic_inv, 0.0)
    qc1N, qc1Ncs = smt.get_qc1N_qc1Ncs(CPT_qt, CPT_fs, sigmav, sigmavp, FC)
    qc1N_inv, qc1Ncs_inv = smt.get_qc1N_qc1Ncs(qt_inv, fs_inv, sigmav, sigmavp, FC)
    # ztop, zbot, qc1Ncs_lay, Ic_lay = smt.cpt_layering(qc1Ncs_inv, Ic_inv, CPT_depth, dGWT=dGWT, Nmin = 1, Nmax = None, averaging = 1)
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

    # Clean sand correction (Robertson 2021)
    # index1= (Ic_lay_resampled > 1.7) & (Ic_lay_resampled < 3.0)
    # index2 = (Ic_lay_resampled >= 3.0)
    Qtncs = Qtn_lay_resampled.copy()
    kc = np.full(Qtn_lay_resampled.shape, 1.0)

    # kc[index1] = 1.8346 * Ic_lay_resampled[index1] ** 5 - 23.673 * Ic_lay_resampled[index1] ** 4 + 124.02 * Ic_lay_resampled[index1] ** 3 - 320.616 * Ic_lay_resampled[index1] ** 2 + 405.821 * Ic_lay_resampled[index1] - 199.97
    # kc[index2] = 1.8346 * 3.0 ** 5 - 23.673 * 3.0 ** 4 + 124.02 * 3.0 ** 3 - 320.616 * 3.0 ** 2 + 405.821 * 3.0 - 199.97

    kc = 1 + 5 / (1 + np.exp( - Kc_c1 * (Ic_lay_resampled - Kc_c2) ))

    Qtncs = Qtn_lay_resampled * kc

    return fz, Ic_lay_resampled, Qtncs, CPT_depth, Qtn_inv, Ic_inv

def CPT_Model_to_Vs (travelTimeMeta_ID, x): 
    # fz, Ic, Qtncs, CPT_depth, Qtn_inv, Ic_inv = get_CPT_KC(travelTimeMeta_ID, Kc_c1, Kc_c2)
    fz, Ic, Qtncs, CPT_depth, Qtn_inv, Ic_inv = get_CPT(travelTimeMeta_ID)

    valid_indices = (fz > 0) & (Qtncs > 0) & (Ic >= 1)

    Qtncs = np.asarray(Qtncs[valid_indices])
    Ic = np.asarray(Ic[valid_indices])
    fz = np.asarray(fz[valid_indices])
    CPT_depth = np.asarray(CPT_depth[valid_indices])
    Qtn_inv = np.asarray(Qtn_inv[valid_indices])
    Ic_inv = np.asarray(Ic_inv[valid_indices])

    # f_Ic = 1 / (1 + np.exp(-10 * (Ic - 2.6)))

    X = np.column_stack((np.log(Qtncs), Ic * np.log(fz), np.log(fz)))
    Vs_comp = x[:-1] @ X.T + x[3]


    tt_id = np.full(len(CPT_depth), travelTimeMeta_ID)

    return tt_id, Vs_comp, CPT_depth, fz, Ic, Qtncs, Qtn_inv, Ic_inv


# def TT_average (travelTimeMeta_ID, Kc_c1, Kc_c2, source = 1.0):
def TT_average (travelTimeMeta_ID, source = 1.0):
    ztop_TT, zbot_TT = Preprocessing_TT(travelTimeMeta_ID, source = source)[-2:]
    # fz, Ic_lay_resampled, Qtncs, CPT_depth, qt_inv, Ic_inv = get_CPT_KC(travelTimeMeta_ID, Kc_c1, Kc_c2)
    fz, Ic_lay_resampled, Qtncs, CPT_depth, qt_inv, Ic_inv = get_CPT(travelTimeMeta_ID)
    
    Qtn_TT = np.zeros(len(ztop_TT))
    Ic_TT = np.zeros(len(ztop_TT))
    fz_TT = np.zeros(len(ztop_TT))
    
    for i in range(len(ztop_TT)):
        Qtn_TT[i] =np.mean(Qtncs[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])])
        Ic_TT[i] = np.mean(Ic_lay_resampled[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])])
        fz_TT[i] = np.mean(fz[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])])

    return Qtn_TT, Ic_TT, fz_TT, ztop_TT, zbot_TT

def CPT_Model_to_Vs_KC (travelTimeMeta_ID, x, Kc_c1, Kc_c2): 
    fz, Ic, Qtncs, CPT_depth, Qtn_inv, Ic_inv = get_CPT_KC(travelTimeMeta_ID, Kc_c1, Kc_c2)
    # fz, Ic, Qtncs, CPT_depth, Qtn_inv, Ic_inv = get_CPT(travelTimeMeta_ID)

    valid_indices = (fz > 0) & (Qtncs > 0) & (Ic >= 1)

    Qtncs = np.asarray(Qtncs[valid_indices])
    Ic = np.asarray(Ic[valid_indices])
    fz = np.asarray(fz[valid_indices])
    CPT_depth = np.asarray(CPT_depth[valid_indices])
    Qtn_inv = np.asarray(Qtn_inv[valid_indices])
    Ic_inv = np.asarray(Ic_inv[valid_indices])

    # f_Ic = 1 / (1 + np.exp(-10 * (Ic - 2.6)))

    X = np.column_stack((np.log(Qtncs), Ic * np.log(fz), np.log(fz)))
    Vs_comp = x[:-1] @ X.T + x[3]

    tt_id = np.full(len(CPT_depth), travelTimeMeta_ID)

    return tt_id, Vs_comp, CPT_depth, fz, Ic, Qtncs, Qtn_inv, Ic_inv


# def TT_average (travelTimeMeta_ID, Kc_c1, Kc_c2, source = 1.0):
def TT_average_KC (travelTimeMeta_ID, Kc_c1, Kc_c2, source = 1.0):
    ztop_TT, zbot_TT = Preprocessing_TT(travelTimeMeta_ID, source = source)[-2:]
    fz, Ic_lay_resampled, Qtncs, CPT_depth, qt_inv, Ic_inv = get_CPT_KC(travelTimeMeta_ID, Kc_c1, Kc_c2)
    # fz, Ic_lay_resampled, Qtncs, CPT_depth, qt_inv, Ic_inv = get_CPT(travelTimeMeta_ID)
    
    Qtn_TT = np.zeros(len(ztop_TT))
    Ic_TT = np.zeros(len(ztop_TT))
    fz_TT = np.zeros(len(ztop_TT))
    
    for i in range(len(ztop_TT)):
        Qtn_TT[i] =np.mean(Qtncs[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])])
        Ic_TT[i] = np.mean(Ic_lay_resampled[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])])
        fz_TT[i] = np.mean(fz[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])])

    return Qtn_TT, Ic_TT, fz_TT, ztop_TT, zbot_TT


def CPT_to_Vs (fz, Qtn_lay_resampled, Ic_lay_resampled, CPT_depth, halfdepth, DTT, ztop_TT, zbot_TT, 
               c0 = 1.93, c1 = 0.5, c2 = 0.25
            #    , c3 = 0.63
               ):

    # Formulation
    # c0 = 1.93
    # c1 = 0.5
    # c2 = 0.25
    # c3 = 0.63

    #Generate the layered Vs profile
    vs_lay = np.zeros(len(CPT_depth))
    slowness_vs_lay = np.zeros(len(CPT_depth))
    for i in range(len(CPT_depth)):
        vs_lay[i] = np.exp(c0 + c1 * np.log(Qtn_lay_resampled[i]) + c2 * (1/(1+np.exp(-(Ic_lay_resampled[i]-2.6)/0.2))+1) * np.log(fz[i]) 
                        #    + c3 * np.log(np.exp(Ic_lay_resampled[i]))
                           )
        slowness_vs_lay[i] = 1/vs_lay[i]

    Depth_interval = []
    slowness_interval = []
    TTfromslowness = np.zeros(len(halfdepth))
    for i in range(len(halfdepth)):
        Depth_interval = CPT_depth[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])]
        slowness_interval = slowness_vs_lay[(CPT_depth >= ztop_TT[i]) & (CPT_depth < zbot_TT[i])]
        TTfromslowness[i] = (np.trapz(slowness_interval, Depth_interval))

    TTfromslowness = TTfromslowness * 1000  # Convert to ms

    # Generate the difference between the two travel time difference (RMSE)
    RMSE = np.sqrt(np.mean((DTT - TTfromslowness)**2))
    MSE = np.mean((DTT - TTfromslowness)**2)
    MAE = np.mean(np.abs(DTT - TTfromslowness))

    Misfit = MSE

    return Misfit, vs_lay, slowness_vs_lay, TTfromslowness

from scipy.optimize import minimize

from scipy.optimize import minimize

# Wrapper function for optimization
def objective(params, fz, Qtn_lay_resampled, Ic_lay_resampled, CPT_depth, halfdepth, DTT, ztop_TT, zbot_TT):
    c0, c1, c2 = params
    Misfit, vs_lay, slowness_vs_lay, TTfromslowness = CPT_to_Vs (fz, Qtn_lay_resampled, Ic_lay_resampled, CPT_depth, halfdepth, DTT, ztop_TT, zbot_TT, c0, c1, c2)
    return Misfit  # Minimize the difference between modeled and actual travel times

def optimization (travelTimeMeta_ID, source = 1.0):
    # Initial guesses for c0, c1, c2 (robertson 2012 parameters)
    initial_params = [1.93, 0.5, 0.25]

    # Bounds for c0, c1, c2, and c3
    # bounds = [(None, None), (-0.91, 1.31), (-0.13, 0.47), (-1.92, 2.22)]
    bounds = [(None, None), (None, None), (None, None)]

    # Fixed parameters
    fz, Qtn_lay_resampled, Ic_lay_resampled, Qtn_lay, Ic_lay, Qtn_inv, Ic_inv, CPT_qt, CPT_depth, ztop, zbot = Preprocessing_CPT(travelTimeMeta_ID)
    TT, TT_depth, halfdepth, DTT, slowness_TT, slowness_TT_depth, ztop_TT, zbot_TT = Preprocessing_TT(travelTimeMeta_ID, source = source)

    # Optimize c0, c1, c2, and c3
    result = minimize(objective, initial_params, args=(fz, Qtn_lay_resampled, Ic_lay_resampled, CPT_depth, halfdepth, DTT, ztop_TT, zbot_TT), 
                    method='L-BFGS-B', bounds=bounds, tol = 1.e-20)  # Use L-BFGS-B method for bounded optimization

    return result.x[0], result.x[1], result.x[2], result.fun

def regression (id, n, source = 1.0):

    tt_meta, tt_data, cpt_meta1, cpt_data1, tt_depth, Meas_TT, cpt_id, cpt_depth1, qt1, fs1 = query_TT(id)

    # Correction of Source Offset Distance
    traveltime = tt_depth / np.sqrt (tt_depth**2 + source**2) * Meas_TT

    # Compute half depths from tt_depth
    half_depths = (tt_depth[:-1] + tt_depth[1:]) / 2

    # joint the half depths with tt_depth (without the first and last point) as candidate breakpoints
    candidate_breakpoints = np.concatenate((tt_depth[1:-1], half_depths))

    n_segments = n

    if n_segments == 1:
        piecewise = pwlf.PiecewiseLinFit(tt_depth, traveltime)
        breaks = piecewise.fit(n_segments)
        fitted_values = piecewise.predict(tt_depth)
        fitted_values_for_plot = fitted_values
        new_depths = tt_depth
    elif n_segments - 1 > len(half_depths):
        raise ValueError("Not enough candidate breakpoints for the number of segments")
    else: 
        best_rss = np.inf
        piecewise = None
        # Generate sorted combinations of candidate breakpoints
        for bp_combo in combinations(sorted(set(candidate_breakpoints)), n_segments - 1):
            # print (f"Trying breakpoints: {bp_combo}")
            try:
                all_breaks = sorted([tt_depth.min()] + list(bp_combo) + [tt_depth.max()])

                # Ensure each segment has at least two points
                for i in range(len(all_breaks) - 1):
                    count = np.sum((tt_depth >= all_breaks[i]) & (tt_depth <= all_breaks[i + 1]))
                    if count < 2:
                        break  # Invalid segment, skip this combo
                else:
                    # Only fits if all segments passed the test
                    piecewise_trial = pwlf.PiecewiseLinFit(tt_depth, traveltime)
                    rss = piecewise_trial.fit_with_breaks(all_breaks)
                    if rss < best_rss:
                        best_rss = rss
                        piecewise = piecewise_trial
                        breaks = all_breaks
            except Exception as e:
                print(f"Error fitting with breaks {bp_combo}: {e}")

        if piecewise is None or 'breaks' not in locals():
            raise ValueError(f"No valid breakpoint combination found for id={id} with n={n_segments}")

        # Combine breaks and tt_depth, then sort them
        new_depths = np.array(sorted(set(breaks + list(tt_depth))))

        # Generate the fitted values
        fitted_values = piecewise.predict(tt_depth)
        fitted_values_for_plot = piecewise.predict(new_depths)
    
    # Calculate RMSE
    residuals = traveltime - fitted_values
    rmse = np.sqrt(np.mean(residuals**2))

    # Calculate slopes of each segment
    slopes = piecewise.slopes
    ztop = breaks[:-1]
    # Replace the first ztop with the first CPT_depth1
    ztop[0] = cpt_depth1[0]
    zbot = breaks[1:]
    
    return {
        'breaks': breaks,
        'candidate_breakpoints': candidate_breakpoints,
        'fitted_values': fitted_values,
        'fitted_values_for_plot': fitted_values_for_plot,
        'new_depths': new_depths,
        'rmse': rmse,
        'slopes': slopes,
        'ztop': ztop,
        'zbot': zbot
    }

def regression_full_manual(id, breakpoints, source = 1.0):
    # Retrieve data
    tt_meta, tt_data, cpt_meta1, cpt_data1, tt_depth, Meas_TT, cpt_id, cpt_depth1, qt1, fs1 = query_TT(id)

    # Correction of Source Offset Distance
    traveltime = tt_depth / np.sqrt (tt_depth**2 + source**2) * Meas_TT

    # Validate breakpoints
    breakpoints = sorted(set(breakpoints))  # Ensure sorted and unique
    if breakpoints[0] <= tt_depth.min() or breakpoints[-1] >= tt_depth.max():
        raise ValueError("Breakpoints must be within the range of tt_depth (excluding endpoints)")

    all_breaks = [tt_depth.min()] + breakpoints + [tt_depth.max()]

    # Ensure each segment has at least two data points
    for i in range(len(all_breaks) - 1):
        count = np.sum((tt_depth >= all_breaks[i]) & (tt_depth <= all_breaks[i + 1]))
        if count < 2:
            raise ValueError(f"Segment between {all_breaks[i]} and {all_breaks[i + 1]} has less than 2 data points.")

    # Fit with custom breakpoints
    piecewise = pwlf.PiecewiseLinFit(tt_depth, traveltime)
    rss = piecewise.fit_with_breaks(all_breaks)

    # Generate fitted values
    new_depths = np.array(sorted(set(all_breaks + list(tt_depth))))
    fitted_values = piecewise.predict(tt_depth)
    fitted_values_for_plot = piecewise.predict(new_depths)

    # RMSE
    residuals = traveltime - fitted_values
    rmse = np.sqrt(np.mean(residuals**2))

    # Slopes
    slopes = piecewise.slopes
    ztop = all_breaks[:-1]
    # Replace the first ztop with the first CPT_depth1
    ztop[0] = cpt_depth1[0]

    zbot = all_breaks[1:]

    return {
        'breaks': all_breaks,
        'fitted_values': fitted_values,
        'fitted_values_for_plot': fitted_values_for_plot,
        'new_depths': new_depths,
        'rmse': rmse,
        'slopes': slopes,
        'ztop': ztop,
        'zbot': zbot
    }


def optimize (id):
    tt_meta, tt_data, cpt_meta1, cpt_data1, tt_depth, Meas_TT, cpt_id, cpt_depth1, qt1, fs1 = query_TT(id)

    if len(tt_depth) < 7: 
        error = np.zeros(len(tt_depth)-1)
        for i in range (1, len(tt_depth)):
            # start_time = time.time()
            results = regression(id, i)
            error[i-1] = results['rmse']
            # end_time = time.time()
            # print(f"Number of segment {i}, RMSE: {error[i-1]:.4f}, Time taken: {end_time - start_time:.2f} seconds")

    else: 
        error = np.zeros(6)

        for i in range (1, 7): 
            # start_time = time.time()
            results = regression(id, i)
            error[i-1] = results['rmse']
            # end_time = time.time()
            # print(f"Number of segment {i}, RMSE: {error[i-1]:.4f}, Time taken: {end_time - start_time:.2f} seconds")

    # Compute the % change of error between each segment
    error_change = np.zeros(len(error)-1)
    for i in range (1,len(error_change)):
        error_change[i-1] = abs((error[i] - error[i-1]))/error[0] * 100
        if error[i-1] == 0:
            error_change[i-1] = 0

    # Get the number of segment when change of error is less than 10% 
    for i in range (1,len(error_change)+1):
        if error[i] < 0.5:
            if error_change[i-1] < 10 or error[i] <= 0.1:
                break
    Optimum = i+1
    return Optimum
    

def optimized_regression(id, source = 1.0):
    n = optimize(id)
    results = regression(id, n, source = source)

    velocity = (1/np.array(results['slopes']))*1000

    return velocity, results

def Master (travelTimeMeta_ID, c0 = 1.93, c1 = 0.5, c2 = 0.25, slope_break_method = 1, breakpoints = None, source = 1.0
            ):
    fz, Qtn_lay_resampled, Ic_lay_resampled, Qtn_lay, Ic_lay, Qtn_inv, Ic_inv, CPT_qt, CPT_depth, ztop, zbot = Preprocessing_CPT(travelTimeMeta_ID)
    TT, TT_depth, halfdepth, DTT, slowness_TT, slowness_TT_depth, ztop_TT, zbot_TT = Preprocessing_TT(travelTimeMeta_ID, source = source)
    Misfit, vs_lay, slowness_vs_lay, TTfromslowness = CPT_to_Vs(fz, Qtn_lay_resampled, Ic_lay_resampled, CPT_depth, halfdepth, DTT, ztop_TT, zbot_TT, 
                                       c0 = c0, c1 = c1, c2 = c2)
    # Full Automatic
    if slope_break_method == 0: 
        velocity, results = optimized_regression(travelTimeMeta_ID, source = source)
    # Full Manual
    elif slope_break_method == 1:
        if breakpoints is None:
            raise ValueError("Breakpoints must be provided for manual regression.")
        results = regression_full_manual(travelTimeMeta_ID, breakpoints, source = source)
        velocity = (1/np.array(results['slopes']))*1000

    top_depth = results['ztop']
    bottom_depth = results['zbot']

    return Qtn_inv, CPT_depth, Qtn_lay, ztop, zbot, Ic_inv, Ic_lay, vs_lay, velocity, TTfromslowness, halfdepth, DTT, TT, TT_depth, results, top_depth, bottom_depth


