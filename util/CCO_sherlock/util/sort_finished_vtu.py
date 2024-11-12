import os
import sys
import math
import numpy as np
import pickle
import shutil
sys.path.append("/home/nrubio/Desktop/SV_scripts") # need to append absolute path to directory where helper_functions.py is stored
from write_solver_files import *


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_random_params(params_base):
    rng = np.random.default_rng(seed=None)
    params_rand = {}
    for param in params_base.keys():
        params_rand.update({param: params_base[param] * (0.8 + rng.random()*(1.2-0.8))})
    return params_rand

def get_centerlines():

    home_dir = os.path.expanduser("~")
    dir = home_dir + "/Desktop"
    #dir = "/media/marsdenlab/Data1/synthetic_junctions"
    os.chdir(f"{dir}/synthetic_junctions")
    geos = os.listdir(f"{dir}/synthetic_junctions"); geos.sort()
    os.chdir(dir)
    for geo in geos:

        try:
            os.mkdir(f"{dir}/centerlines/{geo}")
            #shutil.copy(f"{dir}/junction_sim_files/{geo}/centerlines/centerlines/centerline.vtp", f"{dir}/centerlines/{geo}")
            shutil.copy(f"{dir}/synthetic_junctions/{geo}/centerlines/centerline.vtp", f"{dir}/centerlines/{geo}")
        except:
            continue
        # for i in range(10):
        #     try:
        #         shutil.move(f"{dir}/{geo}/flow_{i}_avg_sol", f"{home_dir}/Desktop/synthetic_junction_results/{geo}")
        #     except:
        #         continue
    return

def get_results():

    home_dir = os.path.expanduser("~")
    dir = home_dir + "/Desktop/junction_sim_files"
    #dir = "/media/marsdenlab/Data1/synthetic_junctions"
    os.chdir(dir)
    geos = os.listdir(); geos.sort()
    for geo in geos:
        os.chdir(dir)
        os.chdir(geo)
        if os.path.exists(f"{home_dir}/Desktop/synthetic_junction_results/{geo}") == False:
            os.mkdir(f"{home_dir}/Desktop/synthetic_junction_results/{geo}")
        try:
            shutil.move(f"{home_dir}/Desktop/synthetic_junction_results/{geo}", f"{home_dir}/Desktop/synthetic_junction_results/{geo}")
        except:
            continue
        # for i in range(10):
        #     try:
        #         shutil.move(f"{dir}/{geo}/flow_{i}_avg_sol", f"{home_dir}/Desktop/synthetic_junction_results/{geo}")
        #     except:
        #         continue
    return

def move_geos():

    home_dir = os.path.expanduser("~")
    source_dir = "/media/marsdenlab/Data1/synthetic_junctions"
    dir = home_dir + "/Desktop/synthetic_junctions"
    #dir = "/media/marsdenlab/Data1/synthetic_junctions"
    os.chdir(dir)
    geos = os.listdir("/media/marsdenlab/Data1/synthetic_junctions"); geos.sort()
    for geo in geos[135:]:
        #os.chdir(dir)
        #os.chdir(geo)
        if os.path.exists(f"{dir}/{geo}") == False:
            os.mkdir(f"{dir}/{geo}")
        #try:
        shutil.copytree(f"{source_dir}/{geo}/mesh-complete", f"{dir}/{geo}/mesh-complete")
        shutil.copy(f"{source_dir}/{geo}/geo_params", f"{dir}/{geo}/geo_params")
        print(geo)
        # except:
        #     #print(geo)
        #     # try:
        #     #     shutil.move(f"{home_dir}/Desktop/junction_sim_files/{geo}/mesh-complete", f"{dir}/{geo}")
        #     #     shutil.move(f"{home_dir}/Desktop/junction_sim_files/{geo}/geo_params", f"{dir}/{geo}")
        #     # except:
        #     continue
        #     continue
        # for i in range(10):
        #     try:
        #         shutil.move(f"{dir}/{geo}/flow_{i}_avg_sol", f"{home_dir}/Desktop/synthetic_junction_results/{geo}")
        #     except:
        #         continue
    return

def get_finished_flow_vtu():

    home_dir = os.path.expanduser("~")
    dir = home_dir + "/Desktop/synthetic_junctions"
    #dir = "/media/marsdenlab/Data1/synthetic_junctions"
    os.chdir(dir)
    geos = os.listdir(); geos.sort()
    for geo in geos:
        os.chdir(dir)
        if os.path.exists(f"{dir}/{geo}/geo_params") == False:
            print(geo)
            import pdb; pdb.set_trace()
            shutil.rmtree(geo)
            continue
        os.chdir(geo)

        if os.path.exists(dir+"/"+geo+"/"+"solution_flow_0.vtu"):
            print(f"moving {geo}")
            shutil.move(dir+"/"+geo, home_dir + "/Desktop/junction_sim_files")
        else:
            for i in range(10):
                try:
                    os.remove(dir+"/"+geo+"/"+"flow_%d_avg_sol"%i)
                except:
                    continue

    return

if __name__ == "__main__":
    #get_finished_flow_vtu()
    #get_results()
    get_centerlines()
    #move_geos()
