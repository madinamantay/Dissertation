import argparse
import os
import shutil

from gan import load_dataset, get_generator, get_discriminator, GAN, plot_metrics


parser = argparse.ArgumentParser(description='GAN model')
parser.add_argument('--action', type=str, required=True, help='Action to perform')
parser.add_argument('--data-dir', type=str, help='Data directory')
parser.add_argument('--cp-dir', type=str, help='Checkpoint directory')
parser.add_argument('--out-dir', type=str, help='Directory for output files')
parser.add_argument('--cp-path', type=str, help='Path to checkpoint to load')
parser.add_argument('--count', type=int, help='Count of output image for each class')
parser.add_argument('--report-file', type=str, help='Path to report file with metrics of losses')
parser.add_argument('--output-file', type=str, help='Path of output file')
parser.add_argument('--source-dir-1', type=str, help='Path to dataset')
parser.add_argument('--source-dir-2', type=str, help='Path to dataset')
parser.add_argument('--target-dir', type=str, help='Path of target dataset')


def main(args):
    match args.action:
        case 'train':
            train(args.data_dir, args.cp_dir, args.out_dir, args.report_file)
        case 'generate':
            generate(args.cp_path, args.out_dir, args.count)
        case 'plot_metrics':
            plot_metrics(args.report_file, args.output_file)
        case 'mix_datasets':
            mix_datasets(args.source_dir_1, args.source_dir_2, args.target_dir)


def train(data_dir: str, cp_dir: str, out_dir: str, report_file: str):
    gan = GAN()

    dataset = load_dataset(data_dir)
    gan.set_dataset(dataset)

    gen = get_generator()
    disc = get_discriminator()
    gan.set_models(gen, disc)

    gan.train(cp_dir, out_dir)
    gan.import_losses(report_file)


def generate(cp_path: str, out_dir: str, count: int):
    gan = GAN()

    gen = get_generator()
    disc = get_discriminator()
    gan.set_models(gen, disc)

    gan.load_models(cp_path)

    gan.generate_all(count, out_dir)


def mix_datasets(source1: str, source2: str, target: str):
    if not os.path.exists(target):
        os.makedirs(target)
  
    shutil.copytree(source1, target, dirs_exist_ok=True)
    shutil.copytree(source2, target, dirs_exist_ok=True)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
