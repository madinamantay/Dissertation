import argparse

from classes import class_names
from classifier import load_train_dataset, load_test_dataset, get_classifier_model, Classifier, plot_metrics


parser = argparse.ArgumentParser(description='Classifier model')
parser.add_argument('--action', type=str, required=True, help='Action to perform')
parser.add_argument('--train-dir', type=str, help='Train data directory')
parser.add_argument('--test-dir', type=str, help='Test data directory')
parser.add_argument('--report-file', type=str, help='Path to report file with metrics of accuracies and losses')
parser.add_argument('--model-path', type=str, help='Path to model')
parser.add_argument('--image-path', type=str, help='Path to image to test')
parser.add_argument('--output-file', type=str, help='Path of output file')


def main(args):
    match args.action:
        case 'train':
            train(args.train_dir, args.model_path, args.report_file)
        case 'test':
            test(args.model_path, args.test_dir)
        case 'test_once':
            test_once(args.model_path, args.image_path)
        case 'plot_metrics':
            plot_metrics(args.report_file, args.output_file)


def train(train_dir: str, model_path: str, report_file: str):
    cl = Classifier()

    x, y = load_train_dataset(train_dir)
    cl.set_train_data(x, y)

    model = get_classifier_model()
    cl.set_model(model)

    cl.train(model_path)
    cl.import_metrics(report_file)

    print('Train finished')


def test(model_path: str, test_dir: str):
    cl = Classifier()

    model = get_classifier_model()
    cl.set_model(model)
    cl.load_model(model_path)

    x, y = load_test_dataset(test_dir)
    cl.set_test_data(x, y)

    test_acc = cl.test()
    print(f'Test accuracy: {test_acc}')


def test_once(model_path: str, image_path: str):
    cl = Classifier()

    model = get_classifier_model()
    cl.set_model(model)
    cl.load_model(model_path)

    predicted_class = cl.test_once(image_path)
    class_name = class_names[predicted_class]

    print(f'Predicted image class: {class_name} (class_label: {predicted_class})')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
