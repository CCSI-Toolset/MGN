import os
from sys import path

#########
# local application imports
# get path to root of the project
mgn_code_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
path.append(mgn_code_dir)


from GNN.utils.rollout import build_parser_full_domains, main_full_domains

"""
example usage:
    python rollout_full_domain.py config_files/test_pnnl_subdomain.ini --cpu --viz_rollout_num=1 --GT
"""

if __name__ == "__main__":
    parser = build_parser_full_domains()
    args = parser.parse_args()
    main_full_domains(args)
