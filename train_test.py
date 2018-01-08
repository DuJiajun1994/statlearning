import sys
import argparse
import pickle
import pandas as pd
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
    train_data, train_label = provider.get_data('trainval')
    test_data, test_data_name = provider.get_test_data()
    model = train_model(model_name, train_data, train_label, cfg)
    predict = eval_model(model, test_data)

    df = pd.DataFrame(columns=['image_name', 'label'])
    result_path = '{}_test_result.csv'.format(model_name)
    for i in range(len(test_data_name)):
        print(i)
        df.loc[i] = [test_data_name[i], predict[i]]
    df.to_csv(result_path, index=False)
