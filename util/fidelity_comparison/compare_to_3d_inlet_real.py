from cProfile import label
from cmath import phase
import sys
from turtle import color, down
from matplotlib import lines
from matplotlib.pylab import f
import pandas as pd
from pyparsing import line
from scipy.fftpack import shift
import vtk
import os
import numpy as np
import pdb
from scipy.interpolate import interp1d

sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm

from util.tools.get_bc_integrals import get_res_names
from util.tools.junction_proc import *
from util.tools.basic import compute_rmse
#from geo_processing import *
from util.tools.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex']=True
colors = ["royalblue", "orangered", "seagreen", "peru", "blueviolet"]


def compare_to_3d_inlet_real(junction_mode = "standard",
                  tree_name = "tree_20_real",
                  time_step = "700"):

    
    tree_name_split = tree_name.split("_")[0:2]
    tree_name_base = "_".join(tree_name_split)

    reader_0d = read_geo(f"trees/zerod_output_cent/standard/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_rri = read_geo(f"trees/zerod_output_cent/RRI/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_ri = read_geo(f"trees/zerod_output_cent/RI/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_3d = read_geo(f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_real.vtp").GetOutput()
    #was 300

    arrays_0d = get_all_arrays(reader_0d)
    arrays_rri = get_all_arrays(reader_rri)
    arrays_ri = get_all_arrays(reader_ri)
    arrays_3d = get_all_arrays(reader_3d)

    
    times_3d = [int(key[9:]) for key in arrays_3d.keys() if "pressure" in key]
    times_3d = [time for time in times_3d if time % 10 == 0]  # Only take every 10th time step
    times_0d = [float(key[9:]) for key in arrays_0d.keys() if "pressure" in key]
    times_rri = [float(key[9:]) for key in arrays_rri.keys() if "pressure" in key]
    times_ri = [float(key[9:]) for key in arrays_ri.keys() if "pressure" in key]

    dt = times_0d[1] - times_0d[0]
    dt_3d = 0.01
    branch0_locs = np.where(arrays_3d["BranchId"] == 0)[0]
    branch0_valid_locs = np.where(~np.isnan(arrays_3d["pressure_100"][branch0_locs])) # Get the first value of the pressure for branch 0
    inlet_gid = arrays_3d["GlobalNodeId"][branch0_locs[branch0_valid_locs[0][0]]]
    # inlet_gid = arrays_3d["GlobalNodeId"][branch0_valid_locs[0][0]]
    # inlet_gid = 23
    flows_0d = []; pressures_0d = []
    for time in times_0d:
        flows_0d.append(arrays_0d[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_0d.append(arrays_0d[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)

    flows_rri = []; pressures_rri = []
    for time in times_rri:
        flows_rri.append(arrays_rri[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_rri.append(arrays_rri[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)

    flows_ri = []; pressures_ri = []
    for time in times_rri:
        flows_ri.append(arrays_ri[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_ri.append(arrays_ri[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)
    # pdb.set_trace()
    flows_3d = []; pressures_3d = []

    for time in times_3d:
        flows_3d.append(arrays_3d[f"velocity_{time:03d}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_3d.append(arrays_3d[f"pressure_{time:03d}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)
    times_3d = [time * dt_3d for time in times_3d]
    
    zipped_lists = zip(times_3d, flows_3d, pressures_3d)
    sorted_zipped_lists = sorted(zipped_lists)
    # Unzip the sorted lists
    times_3d, flows_3d, pressures_3d = zip(*sorted_zipped_lists)
    n_reps = 3
    times_3d_last = times_3d[-len(times_3d)//n_reps:]  # Only take the second half of the time steps
    flows_3d_last = flows_3d[-len(flows_3d)//n_reps:]
    pressures_3d_last = pressures_3d[-len(pressures_3d)//n_reps:]
    
    pdb.set_trace()
    times_3d_lst_fine = np.linspace(times_3d_last[0], times_3d_last[-1], 100)
    flows_3d_last_fine = interp1d(times_3d_last, flows_3d_last, kind='quadratic')(times_3d_lst_fine)
    pressures_3d_last_fine = interp1d(times_3d_last, pressures_3d_last, kind='quadratic')(times_3d_lst_fine)
    
    times_3d_fine = np.linspace(times_3d[0], times_3d[-1], 300)
    flows_3d_fine = interp1d(times_3d, flows_3d, kind='quadratic')(times_3d_fine).tolist()
    pressures_3d_fine = interp1d(times_3d, pressures_3d, kind='quadratic')(times_3d_fine).tolist()
    
    pressures_0d_last = pressures_0d[-len(pressures_0d)//n_reps:]
    flows_0d_last = flows_0d[-len(flows_0d)//n_reps:]
    pressures_rri_last = pressures_rri[-len(pressures_rri)//n_reps:]
    flows_rri_last = flows_rri[-len(flows_rri)//n_reps:]
    pressures_ri_last = pressures_ri[-len(pressures_ri)//n_reps:]
    flows_ri_last = flows_ri[-len(flows_ri)//n_reps:]
    
    

    shift_0d_pts = 185; shift_0d = 1.847
    shift_3d_pts = 185; shift_3d = 1.847
    #pdb.set_trace()
    times_0d = [time - shift_0d for time in times_0d[shift_0d_pts:shift_0d_pts+100]]
    times_rri = [time - shift_0d for time in times_rri[shift_0d_pts:shift_0d_pts+100]]
    times_ri = [time - shift_0d for time in times_ri[shift_0d_pts:shift_0d_pts+100]]
    times_3d = [time - shift_3d for time in times_3d[shift_3d_pts:shift_3d_pts+100]]
    times_3d_fine = [time - shift_0d for time in times_3d_fine[shift_0d_pts:shift_0d_pts+100]]
    
    flows_0d = flows_0d[shift_0d_pts:shift_0d_pts+100];pressures_0d = pressures_0d[shift_0d_pts:shift_0d_pts+100]
    flows_rri = flows_rri[shift_0d_pts:shift_0d_pts+100];pressures_rri = pressures_rri[shift_0d_pts:shift_0d_pts+100]
    flows_ri = flows_ri[shift_0d_pts:shift_0d_pts+100];pressures_ri = pressures_ri[shift_0d_pts:shift_0d_pts+100]
    flows_3d = flows_3d[shift_3d_pts:shift_3d_pts+100];pressures_3d = pressures_3d[shift_3d_pts:shift_3d_pts+100]
    flows_3d_fine = flows_3d_fine[shift_0d_pts:shift_0d_pts+100]; pressures_3d_fine = pressures_3d_fine[shift_0d_pts:shift_0d_pts+100]
    
    num_pts = len(times_0d)
    n_reps = 1
    for i in range(n_reps-1):
        times_0d = times_0d + [t + times_0d[-1] for t in times_0d[0:num_pts]]
        times_3d_fine = times_3d_fine + [t + times_3d_fine[-1] for t in times_3d_fine[0:num_pts]]
        times_rri = times_rri + [t + times_rri[-1] for t in times_rri[0:num_pts]]
        times_ri = times_ri + [t + times_ri[-1] for t in times_ri[0:num_pts]]
    flows_0d = n_reps * flows_0d; pressures_0d = n_reps * pressures_0d
    flows_3d_fine = n_reps * flows_3d_fine; pressures_3d_fine = n_reps * pressures_3d_fine
    flows_rri = n_reps * flows_rri; pressures_rri = n_reps * pressures_rri
    flows_ri = n_reps * flows_ri; pressures_ri = n_reps * pressures_ri

    from scipy.signal import windows
    # pressures_rri = 1+np.sin( np.array(times_0d)*2*np.pi)
    # flows_rri = 1+np.sin( np.array(times_0d)*2*np.pi)
    window_reps = 1
    window = windows.tukey(int(len(pressures_0d)/window_reps), alpha = 0.2).tolist() # Create a Hann window for smoothing
    periodic_window = np.asarray(window_reps * window)

    # --------------------- PRESSURE PLOT ---------------------
    plt.clf()
    plt.plot(times_0d, pressures_0d, label="0D standard", color="tomato")
    plt.plot(times_rri, pressures_rri, label="0D RRI", color="seagreen")
    plt.plot(times_ri, pressures_ri, label="0D RI", color="royalblue")
    plt.plot(times_3d_fine, pressures_3d_fine, color="slategrey", label="3D")
    plt.plot(times_0d, pressures_0d*periodic_window, label="0D standard", color="tomato", linestyle = "--")
    plt.plot(times_rri, pressures_rri*periodic_window, label="0D RRI", color="seagreen", linestyle = "--")
    plt.plot(times_ri, pressures_ri*periodic_window, label="0D RI", color="royalblue", linestyle = "--")
    plt.plot(times_3d_fine, pressures_3d_fine*periodic_window, color="slategrey", linestyle = "--")
    plt.xlabel("Flow (cm$^3$/s)")
    plt.ylabel("Pressure (mmHg)")
    plt.yscale("symlog")
    #plt.xlim((0, 3))
    plt.legend()
    os.makedirs(f"results/real/{tree_name}", exist_ok=True)
    plt.savefig(f"results/real/{tree_name}/0d_standard_pf.pdf")

    # --------------------- FLOW PLOT ---------------------
    plt.clf()
    plt.plot(times_0d, flows_0d, color="tomato", linewidth=4)
    plt.plot(times_rri, flows_rri, color="seagreen")
    plt.plot(times_ri, flows_ri, color="royalblue")
    plt.plot(times_3d_fine, flows_3d_fine, color="slategrey")
    plt.plot(times_0d, flows_0d*periodic_window, label="0D standard", color="tomato", linestyle = "--")
    plt.plot(times_rri, flows_rri*periodic_window, label="0D RRI", color="seagreen", linestyle = "--")
    plt.plot(times_ri, flows_ri*periodic_window, label="0D RI", color="royalblue", linestyle = "--")
    plt.plot(times_3d_fine, flows_3d_fine*periodic_window, color="slategrey", linestyle = "--")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (cm$^3$/s)")
    #plt.xlim((0, 4))
    plt.savefig(f"results/real/{tree_name}/0d_ft.pdf")

    # --------------------- PRESSURE LOOP ---------------------
    plt.clf()
    fig = plt.figure(figsize=(2.5, 2.5))
    plt.plot(flows_0d_last,         pressures_0d_last,          color="tomato",             dashes=(1, 1),     label="0D standard", linewidth=3)
    plt.plot(flows_rri_last,        pressures_ri_last,          color="cornflowerblue",     linestyle='dashdot',    label="0D RI", linewidth=2)
    plt.plot(flows_rri_last,        pressures_rri_last,         color="limegreen",          linestyle='dashed',     label="0D RRI", linewidth=2)
    
    plt.plot(flows_3d_fine,    pressures_3d_fine,     color="slategray",          linestyle='solid',      label="3D", linewidth=2)
    plt.xlabel("Flow (cm$^3$/s)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncols = 4)
    plt.savefig(f"results/real/{tree_name}/0d_pressure_loop.pdf", bbox_inches='tight')
    
    # --------------------- PRESSURE RMSE ---------------------
    
    downsample_factor = 50
    dt = 0.001 * downsample_factor
    pressures_0d = np.asarray(pressures_0d)
    pressures_rri = np.asarray(pressures_rri)
    pressures_ri = np.asarray(pressures_ri)
    pressures_3d_fine = np.asarray(pressures_3d_fine)
    flows_0d = np.asarray(flows_0d)
    flows_rri = np.asarray(flows_rri)
    flows_ri = np.asarray(flows_ri)
    flows_3d_fine = np.asarray(flows_3d_fine)
    pressures_0d = pressures_0d[::downsample_factor]
    pressures_rri = pressures_rri[::downsample_factor]
    pressures_ri = pressures_ri[::downsample_factor]
    pressures_3d_fine = pressures_3d_fine[::downsample_factor]
    flows_0d = flows_0d[::downsample_factor]
    flows_rri = flows_rri[::downsample_factor]
    flows_ri = flows_ri[::downsample_factor]
    flows_3d_fine = flows_3d_fine[::downsample_factor]
    

    # pressures_rri = 1+np.sin( np.array(times_0d)*2*np.pi)
    # flows_rri = 1+np.sin( np.array(times_0d)*2*np.pi)
    window = windows.tukey(int(len(pressures_0d)/window_reps), alpha = 0.2).tolist() # Create a Hann window for smoothing
    periodic_window = np.asarray(window_reps * window)#*0+1
    # pressures_windowed = pressures_0d #* window
    # flows_windowed = flows_0d# * window

    # P_fft_0d = np.fft.fft(pressures_0d*periodic_window )
    # P_fft_ri = np.fft.fft(pressures_ri*periodic_window )
    # P_fft_rri = np.fft.fft(pressures_rri*periodic_window )
    # P_fft_3d = np.fft.fft(pressures_3d_fine*periodic_window )
    # Q_fft_0d = np.fft.fft(flows_0d*periodic_window )
    # Q_fft_ri = np.fft.fft(flows_ri*periodic_window )
    # Q_fft_rri = np.fft.fft(flows_rri*periodic_window )
    # Q_fft_3d = np.fft.fft(flows_3d_fine*periodic_window )
    pad_width = int(len(pressures_0d) *1)
    P_fft_0d = np.fft.fft(np.pad(pressures_0d*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    P_fft_ri = np.fft.fft(np.pad(pressures_ri*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    P_fft_rri = np.fft.fft(np.pad(pressures_rri*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    P_fft_3d = np.fft.fft(np.pad(pressures_3d_fine*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    Q_fft_0d = np.fft.fft(np.pad(flows_0d*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    Q_fft_ri = np.fft.fft(np.pad(flows_ri*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    Q_fft_rri = np.fft.fft(np.pad(flows_rri*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    Q_fft_3d = np.fft.fft(np.pad(flows_3d_fine*periodic_window, pad_width=pad_width, mode='constant', constant_values=0) )
    
    sampling_freq = 1/dt
    #sampling_freq_3d = 1/dt_3d
    freq = np.fft.fftfreq(len(P_fft_0d), d=dt)
    # freq_3d = np.fft.fftfreq(len(P_fft_3d), d=0.01)

    N = len(pressures_0d)
    N_3d = len(pressures_3d_fine)
    assert N == N_3d, "Flow lengths do not match"
    plt.clf()
    plt.plot(freq[:int(N//30)], np.abs(P_fft_0d[:int(N//30)]), label = "0D standard", color = "tomato")  # magnitude
    plt.plot(freq[:int(N//30)], np.abs(P_fft_rri[:int(N//30)]), label = "0D RRI", color = "limegreen")  # magnitude
    plt.plot(freq[:int(N//30)], np.abs(P_fft_ri[:int(N//30)]), label = "0D RI", color = "cornflowerblue")  # magnitude
    plt.plot(freq[:int(N//30)], np.abs(P_fft_3d[:int(N//30)]), label = "3D", color = "slategray")  # magnitude
    plt.title("Pressure Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|Z(f)|")
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncols = 4)
    #plt.xscale('log')
    plt.grid(True)
    plt.savefig(f"results/real/{tree_name}/0d_pressure_frequency_domain.pdf", bbox_inches='tight')

    plt.clf()
    plt.plot(freq[:int(N//30)], np.abs(Q_fft_0d[:int(N//30)]), label = "0D standard", color = "tomato")  # magnitude
    plt.plot(freq[:int(N//30)], np.abs(Q_fft_rri[:int(N//30)]), label = "0D RRI", color = "limegreen")  # magnitude
    plt.plot(freq[:int(N//30)], np.abs(Q_fft_ri[:int(N//30)]), label = "0D RI", color = "cornflowerblue")  # magnitude
    plt.plot(freq[:int(N//30)], np.abs(Q_fft_3d[:int(N//30)]), label = "3D", color = "slategray")  # magnitude
    plt.title("Flow Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|Z(f)|")
    plt.legend()
    #plt.xscale('log')
    plt.grid(True)
    plt.savefig(f"results/real/{tree_name}/0d_flow_frequency_domain.pdf", bbox_inches='tight')

    
        
    # Q_f_safe = Q_fft_0d.copy()
    threshold = 0.0000001 # Adjust as needed for your signal's magnitude

    # # from scipy.signal import csd
    # # pdb.set_trace()
    # # f, S_pq = csd(pressures, flow, fs=fs, nperseg=len(pressure))
    # # _, S_qq = csd(flow, flow, fs=fs, nperseg=len(flow))

    # # Z_cross = S_pq / S_qq
    
    # # Option 1: Mask out near-zero frequency components
    mask = np.abs(Q_fft_0d) > threshold;Z_0d = np.full_like(Q_fft_0d, np.nan, dtype=complex)
    Z_0d[mask] = P_fft_0d[mask] / Q_fft_0d[mask]
    #Z_0d = P_fft_0d / Q_fft_0d
    mask_rri = np.abs(Q_fft_rri) > threshold;Z_rri = np.full_like(Q_fft_rri, np.nan, dtype=complex)
    Z_rri[mask_rri] = P_fft_rri[mask_rri] / Q_fft_rri[mask_rri]
    mask_ri = np.abs(Q_fft_ri) > threshold;Z_ri = np.full_like(Q_fft_ri, np.nan, dtype=complex)
    Z_ri[mask_ri] = P_fft_ri[mask_ri] / Q_fft_ri[mask_ri]
    mask_3d = np.abs(Q_fft_3d) > threshold;Z_3d = np.full_like(Q_fft_3d, np.nan, dtype=complex)
    Z_3d[mask_3d] = P_fft_3d[mask_3d] / Q_fft_3d[mask_3d]
    # Z_ri = P_fft_ri / Q_fft_ri
    # Z_rri = P_fft_rri / Q_fft_rri
    # Z_3d = P_fft_3d / Q_fft_3d
    
    valid = ~np.isnan(Z_0d[:int(N//2)] )
    #Z_0d = P_fft_0d / Q_fft_0d
    
    plt.clf()
    plt.plot(freq[:int(N//2)], np.abs(Z_0d[:int(N//2)]), label = "0D standard", color = "tomato")  # magnitude
    plt.plot(freq[:int(N//2)], np.abs(Z_rri[:int(N//2)]), label = "0D RRI", color = "limegreen")  # magnitude
    plt.plot(freq[:int(N//2)], np.abs(Z_ri[:int(N//2)]), label = "0D RI", color = "cornflowerblue")  # magnitude
    plt.plot(freq[:int(N//2)], np.abs(Z_3d[:int(N//2)]), label = "3D", color = "slategray")  # magnitude
    #plt.plot(freq[:int(N//2)][valid], np.abs(Z_0d[:int(N//2)][valid]))  # magnitude
    #plt.plot(freq, np.abs(Z_0d), linestyle='dashed', color='gray')  # full spectrum
    plt.title("Impedance Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|Z(f)|")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/real/{tree_name}/0d_impedance_magnitude.pdf", bbox_inches='tight')
    
    
    valid_0d = ~np.isnan(Z_0d)
    valid_3d = ~np.isnan(Z_3d)
    valid_rri = ~np.isnan(Z_rri)
    valid_ri = ~np.isnan(Z_ri)
    
    freq_valid_0d = freq[valid_0d]
    freq_valid_3d = freq[valid_3d]
    freq_valid_rri = freq[valid_rri]
    freq_valid_ri = freq[valid_ri]
    
    Z_valid_0d = Z_0d[valid_0d]
    Z_valid_3d = Z_3d[valid_3d]
    Z_valid_rri = Z_rri[valid_rri]
    Z_valid_ri = Z_ri[valid_ri]
    

    # Only plot positive frequencies

    #pos_3d = freq_valid_3d >= 0
    f_plot_0d = freq_valid_0d[freq_valid_0d >= 0]
    f_plot_3d = freq_valid_3d[freq_valid_3d >= 0]
    f_plot_rri = freq_valid_rri[freq_valid_rri >= 0]
    f_plot_ri = freq_valid_ri[freq_valid_ri >= 0]
    
    Z_plot_0d = Z_valid_0d[freq_valid_0d >= 0]
    Z_plot_3d = Z_valid_3d[freq_valid_3d >= 0]
    Z_plot_rri = Z_valid_rri[freq_valid_rri >= 0]
    Z_plot_ri = Z_valid_ri[freq_valid_ri >= 0]
    
    # Compute magnitude (in dB) and phase (in degrees)
    #magnitude_db = 20 * np.log10(np.abs(Z_plot))
    mag_0d = np.abs(Z_plot_0d)
    mag_rri = np.abs(Z_plot_rri)
    mag_ri = np.abs(Z_plot_ri)
    mag_3d = np.abs(Z_plot_3d)
    phase_0d = np.angle(Z_plot_0d, deg=True)
    phase_rri = np.angle(Z_plot_rri, deg=True)
    phase_ri = np.angle(Z_plot_ri, deg=True)
    phase_3d = np.angle(Z_plot_3d, deg=True)
    # Bode plot
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # ax_mag.semilogx(f_plot_0d, mag_0d, label="0D standard", color="tomato")
    # ax_mag.semilogx(f_plot_rri, mag_rri, label="0D RRI", color="limegreen")
    # ax_mag.semilogx(f_plot_ri, mag_ri, label="0D RI", color="cornflowerblue")
    # ax_mag.semilogx(f_plot_3d, mag_3d, label="3D", color="slategray")
    ax_mag.plot(f_plot_0d, mag_0d, label="0D standard", color="tomato")
    ax_mag.plot(f_plot_rri, mag_rri, label="0D RRI", color="limegreen")
    ax_mag.plot(f_plot_ri, mag_ri, label="0D RI", color="cornflowerblue")
    ax_mag.plot(f_plot_3d, mag_3d, label="3D", color="slategray")
    ax_mag.set_ylabel("$|Z(\omega)|$ (mmHg cm$^{-3}$s)\n")
    #ax_mag.set_title("Impedance Frequency Response")
    ax_mag.grid(True, which='both')

    # ax_phase.semilogx(f_plot_0d, phase_0d, label="0D standard", color="tomato")
    # ax_phase.semilogx(f_plot_rri, phase_rri, label="0D RRI", color="limegreen")
    # ax_phase.semilogx(f_plot_ri, phase_ri, label="0D RI", color="cornflowerblue")
    # ax_phase.semilogx(f_plot_3d, phase_3d, label="3D", color="slategray")
    ax_phase.plot(f_plot_0d, phase_0d, label="0D standard", color="tomato")
    ax_phase.plot(f_plot_rri, phase_rri, label="0D RRI", color="limegreen")
    ax_phase.plot(f_plot_ri, phase_ri, label="0D RI", color="cornflowerblue")
    ax_phase.plot(f_plot_3d, phase_3d, label="3D", color="slategray")
    ax_phase.set_xlabel("Frequency: $\omega$ (Hz)")
    plt.rcParams['text.usetex'] = True
    ax_phase.set_ylabel(r"$\angle$ $ Z(\omega)$ (degrees)")
    #ax_phase.set_ylabel(r"Phase  \n \angle $ Z(\omega)$ (degrees)")
    ax_phase.grid(True, which='both')

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0.5, -0.5), loc='lower center', ncols = 4,frameon =False)
    plt.savefig(f"results/real/{tree_name}/0d_impedance_bode_plot.pdf", bbox_inches='tight')
    pdb.set_trace()
    return

if __name__ == "__main__":
    compare_to_3d_inlet_real(junction_mode = sys.argv[1], tree_name= sys.argv[2])
                  