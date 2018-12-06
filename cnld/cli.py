## cnld / cli.py


import argparse
import importlib

scripts = {}
scripts['imp-resp-db'] = 'cnld.scripts.imp_resp_db'
scripts['kmat-comsol'] = 'cnld.scripts.kmat_comsol'
scripts['kmat-pzflex'] = 'cnld.scripts.kmat_pzflex'
scripts['kmat-square-mesh-file'] = 'cnld.scripts.kmat_square_mesh_file'

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('script_name')
parser.set_defaults(lookup=scripts)

def main():
    args, unknown_args = parser.parse_known_args()
    args.lookup[args.script_name].main(unknown_args)
    # print(args)
    # print(unknown_args)

    # script_name = args.pop('script_name')
    # lookup = args.pop('lookup')

