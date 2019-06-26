## cnld / cli.py

import argparse
import subprocess


scripts = {}
scripts['matrix-array'] = 'cnld.scripts.matrix_array'
scripts['hex-array'] = 'cnld.scripts.hex_array'
scripts['generate-db'] = 'cnld.scripts.generate_db'

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('script_name')
parser.set_defaults(lookup=scripts)


def main():
    args, unknown_args = parser.parse_known_args()
    subprocess.call(['python', '-m', args.lookup[args.script_name]] + unknown_args)

