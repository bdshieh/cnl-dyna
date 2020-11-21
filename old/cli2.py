'''
Command line interface.
'''
import argparse
import subprocess
import sys

import cnld.scripts

# scripts = {}
# scripts['matrix-array'] = 'cnld.scripts.matrix_array'
# scripts['hex-array'] = 'cnld.scripts.hex_array'
# scripts['generate-db'] = 'cnld.scripts.generate_db'
# scripts['visualize-db'] = 'cnld.scripts.visualize_db'
# scripts['visualize-array'] = 'cnld.scripts.visualize_array'


# run
def run_script(args):

    if args.show_config:
        print(args.Config())
        return

    if args.generate_config:
        abstract.dump(args.Config(), args.generate_config)
        return

    if args.file:
        if args.config:
            cfg = args.Config(**abstract.load(args.config))
        else:
            cfg = args.Config()

        return main(cfg, args)


# define main parser
# parser = argparse.ArgumentParser()
# parser.add_argument('script_name')
# parser.set_defaults(lookup=scripts)
# subparser = parser.add_subparsers()

# define script subparser
scriptparser = argparse.ArgumentParser()
scriptparser.add_argument('-g', '--generate-config')
scriptparser.add_argument('-s', '--show-config', action='store_true')
scriptparser.add_argument('file', nargs='?')
scriptparser.add_argument('-c', '--config')
scriptparser.add_argument('-t', '--threads', type=int)
scriptparser.add_argument('-w', '--write-over', action='store_true')


class Cnld(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Pretends to be git',
                                         usage='''cnld <command> [<args>]

Commands:
   matrix_array     help
   generate_db      asdf
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def matrix_array(self):
        scriptparser.set_defaults(func=run_script,
                                  main=cnld.scripts.matrix_array.main,
                                  Config=cnld.scripts.matrix_array.Config)
        args = scriptparser.parse_args(sys.argv[2:])
        args.func(args)

    def generate_db(self):
        scriptparser.set_defaults(func=run_script,
                                  main=cnld.scripts.generate_db.main,
                                  Config=cnld.scripts.generate_db.Config)
        args = scriptparser.parse_args(sys.argv[2:])
        args.func(args)


def main():
    Cnld()
