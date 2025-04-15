import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff


def cluster_r_quad(anatomy, set_type, require4):
    colors = ["green", "red", "blue"]
    CCO_data_dict = load_dict(f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    offset_number = 60
    for geo_name, geo_dict in CCO_data_dict.items():
        try:
            
            res1 = geo_dict[f"offset_{offset_number}"]["daughter1_R_quad"]
            res2 = geo_dict[f"offset_{offset_number}"]["daughter2_R_quad"]

            if res1 > 0 and res2 > 0:
                color = "green"
            elif res1 < 0 and res2 < 0:
                color = "blue"
            else:
                color = "red"

            area_ratio1 = geo_dict[f"offset_{offset_number}"]["daughter1_area_ratio"]
            area_ratio2 = geo_dict[f"offset_{offset_number}"]["daughter2_area_ratio"]
            #pdb.set_trace()
            plt.scatter(area_ratio1, area_ratio2, color = color)
        except:
            print(f"Error with {geo_name}")
    plt.xlabel("Area Ratio 1")
    plt.ylabel("Area Ratio 2")
    area_ratio1 = np.linspace(0.5, 0.9, 10)
    area_ratio2_const = 1 - area_ratio1
    plt.plot(area_ratio1, area_ratio2_const, color = "black")
    area_ratio2_murray = 1 - area_ratio1**3
    plt.plot(area_ratio1, area_ratio2_murray, color = "black", linestyle = "--")
    if not os.path.exists("results/clustering/"):
        os.makedirs("results/clustering/")
    plt.savefig(f"results/clustering/{anatomy}_{set_type}_cluster_r_quad.png")
    return

if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    require4 = False
    cluster_r_quad(anatomy, set_type, require4)