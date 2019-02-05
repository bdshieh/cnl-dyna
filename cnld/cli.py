## cnld / cli.py

import argparse
import subprocess


scripts = {}
scripts['matrix'] = 'cnld.arrays.matrix'
scripts['freq-resp-db'] = 'cnld.scripts.freq_resp_db'
scripts['imp-resp-db'] = 'cnld.scripts.imp_resp_db'

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('script_name')
parser.set_defaults(lookup=scripts)


def main():
    args, unknown_args = parser.parse_known_args()
    subprocess.call(['python', '-m', args.lookup[args.script_name]] + unknown_args)

