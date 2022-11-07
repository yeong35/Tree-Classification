import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

with open(args.input, 'r') as input_file:
    with open(args.output, 'wb') as output_file:
        for line in input_file:
            output_file.write(line.encode('utf8'))