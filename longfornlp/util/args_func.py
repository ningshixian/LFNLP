import os
import argparse
import numpy as np
import requests
import json


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def get_argv():
    parser = ArgumentParser(prog='_apollo_config.py', epilog='如有疑问请联系406959268@qq.com')  # 写入上面定义的帮助信息
    parser.add_argument('-e', '--environment', dest='environment', type=str, help='operating environment',
                        default='sit')
    parser.add_argument('-ap', '--apollo_private_key_path', dest='apollo_private_key_path', type=str, help='apollo_private_key_path',
                        default='/etc/apollo/apollo_private_key')
    parser.add_argument('-at', '--apollo_token', dest='apollo_token', type=str, help='apollo_token')
    parser.add_argument('-ak', '--apollo_key', dest='apollo_key', type=str, help='apollo_key')
    parser.add_argument('-ac', '--apollo_cluster', dest='apollo_cluster', type=str, help='cluster name in apollo',
                        default='sit')
    parser.add_argument('-aaid', '--apollo_app_id', dest='apollo_app_id', type=str, help='app id in apollo',
                        default='aicare')
    parser.add_argument('-auri', '--apollo_uri', dest='apollo_uri', type=str, help='uri in apollo',
                        default='http://apolloconfig.longfor.sit/')
    parser.add_argument('-ans', '--apollo_namespace', dest='apollo_namespace', type=str, help='namespace in apollo')
    parser.add_argument('-mp', '--model_path', dest='model_path', type=str, help='model_path')
    return parser.parse_args()


# ARGS = get_argv()


from dotmap import DotMap


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict
