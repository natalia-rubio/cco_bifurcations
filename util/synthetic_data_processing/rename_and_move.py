import os
import sys

geos = os.listdir(f"data/synthetic_junctions_reduced_results/Pulmo_rand_dec19_copy"); geos.sort()
for geo in geos:
    os.system(f"cp data/synthetic_junctions_reduced_results/Pulmo_rand_dec19_copy/{geo}/flow_1_red_sol data/synthetic_junctions_reduced_results/Pulmo_rand/{geo}/flow_0_red_sol")
    os.system(f"cp data/synthetic_junctions_reduced_results/Pulmo_rand_dec19_copy/{geo}/flow_3_red_sol data/synthetic_junctions_reduced_results/Pulmo_rand/{geo}/flow_2_red_sol")
