import argparse

from classes import class_names
from classifier import (
    load_train_dataset, load_test_dataset,
    get_classifier_model, get_vgg_11, get_res_net_18, get_mobilenet_v2, Classifier,
    plot_metrics,
)


parser = argparse.ArgumentParser(description='Classifier model')
parser.add_argument('--action', type=str, required=True, help='Action to perform')
parser.add_argument('--train-dir', type=str, help='Train data directory')
parser.add_argument('--test-dir', type=str, help='Test data directory')
parser.add_argument('--report-file', type=str, help='Path to report file with metrics of accuracies and losses')
parser.add_argument('--model-name', type=str, help='Name of model.', choices=('', 'vgg11', 'resnet18', 'mobilenet_v2'))
parser.add_argument('--model-path', type=str, help='Path to model')
parser.add_argument('--classes', type=int, help='Number of classes')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--image-path', type=str, help='Path to image to test')
parser.add_argument('--output-file', type=str, help='Path of output file')


def main(args):
    match args.action:
        case 'train':
            train(args.train_dir, args.model_name, args.classes, args.epochs, args.model_path, args.report_file)
        case 'test':
            test(args.model_name, args.model_path, args.classes, args.test_dir)
        case 'test_once':
            test_once(args.model_name, args.model_path, args.classes, args.image_path)
        case 'plot_metrics':
            plot_metrics(args.report_file, args.output_file)


def train(train_dir: str, model_name: str, classes: int, epochs: int, model_path: str, report_file: str):
    cl = Classifier()

    data_sh, model_sh = _get_shapes(model_name)

    x, y = load_train_dataset(
        path=train_dir,
        classes=classes,
        dim=data_sh,
    )
    cl.set_train_data(x, y)

    model = _get_classifier_model(
        model_name=model_name,
        classes=classes,
        shape=model_sh,
    )
    cl.set_model(model)

    cl.train(
        output_model=model_path,
        classes=classes,
        epochs=epochs,
    )
    cl.import_metrics(report_file)

    print('Train finished')


def test(model_name: str, model_path: str, classes: int, test_dir: str):
    cl = Classifier()

    data_sh, model_sh = _get_shapes(model_name)

    model = _get_classifier_model(
        model_name=model_name,
        classes=classes,
        shape=model_sh,
    )
    cl.set_model(model)
    cl.load_model(model_path)

    x, y = load_test_dataset(
        path=test_dir,
        classes=classes,
        dim=data_sh,
    )
    cl.set_test_data(x, y)

    test_acc = cl.test()
    print(f'Test accuracy: {test_acc}')


def test_once(model_name: str, model_path: str, classes: int, image_path: str):
    cl = Classifier()

    data_sh, model_sh = _get_shapes(model_name)

    model = _get_classifier_model(
        model_name=model_name,
        classes=classes,
        shape=model_sh,
    )
    cl.set_model(model)
    cl.load_model(model_path)

    predicted_class = cl.test_once(image_path, data_sh)
    class_name = class_names[predicted_class]

    print(f'Predicted image class: {class_name} (class_label: {predicted_class})')


def _get_shapes(model_name):
    if model_name == '':
        return (32, 32), (32, 32, 3)

    if model_name == 'vgg11':
        return (32, 32), (32, 32, 3)

    if model_name == 'resnet18':
        return (32, 32), (32, 32, 3)

    if model_name == 'mobilenet_v2':
        return (224, 224), (224, 224, 3)

    return (32, 32), (32, 32, 3)


def _get_classifier_model(model_name: str, classes: int, shape: tuple):
    if model_name == '':
        return get_classifier_model(classes, shape)

    if model_name == 'vgg11':
        return get_vgg_11(classes, shape)

    if model_name == 'resnet18':
        return get_res_net_18(classes, shape)

    if model_name == 'mobilenet_v2':
        return get_mobilenet_v2(classes, shape)

    return get_classifier_model(classes, shape)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
