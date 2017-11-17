import sys
import argparse
from data_provider import DataProvider
from config_provider import get_config
from sklearn_model import train_model, eval_model, compute_accuracy


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Cross Validation')
    parser.add_argument('--model', dest='model_name',
                        help='model to use',
                        default='lr', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    model_name = args.model_name
    cfg = get_config(model_name)
    print('Config:')
    print(cfg)
    provider = DataProvider()
    for i in range(5):
        train_data, train_label, val_data, val_label = provider.get_cv_data(i)
        model = train_model(model_name, train_data, train_label, cfg)
        predict = eval_model(model, val_data)
        accuracy = compute_accuracy(predict, val_label)
        print('accuracy: {}'.format(accuracy))
