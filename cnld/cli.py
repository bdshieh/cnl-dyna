## cnld / cli.py

import argparse
import subprocess


scripts = {}
scripts['matrix'] = 'cnld.arrays.matrix'
scripts['freq-resp-db'] = 'cnld.scripts.freq_resp_db'
scripts['kmat-comsol'] = 'cnld.scripts.kmat_comsol'
scripts['kmat-pzflex'] = 'cnld.scripts.kmat_pzflex'
scripts['kmat-square-mesh-file'] = 'cnld.scripts.kmat_square_mesh_file'

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('script_name')
parser.set_defaults(lookup=scripts)


def main():
    args, unknown_args = parser.parse_known_args()
    subprocess.call(['python', '-m', args.lookup[args.script_name]] + unknown_args)

