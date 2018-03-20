import torch
from argparse import ArgumentParser
from pathlib2 import Path


def main(args):
    input_path = Path(args.input)
    with input_path.open('rb') as f:
        model = torch.load(f)

    model = model.cpu()

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + '_cpu' + input_path.suffix)

    with output_path.open('wb') as f:
        torch.save(model, f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to original .t7 file')
    parser.add_argument('-o', '--output', help='Output path')
    args = parser.parse_args()

    main(args)
