import argparse

import reader

exts = set()
exts.add(".java")

parser = argparse.ArgumentParser()
parser.add_argument("-z", "--size", help="chunk size", type=int, default=100)
parser.add_argument("src_dir")
args = parser.parse_args()

filename = reader.get_filenames(exts, args.src_dir)[0]
print(reader.get_file(filename))
print(args.size)
