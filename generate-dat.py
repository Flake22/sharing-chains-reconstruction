import argparse
from src.data import *

def get_datadist(args, mapping_file=None):
    if args.db=='controlled':
        return Controlled_Dataset(mapping_file, args)
        
    if args.db == 'uncontrolled':
        return Uncontrolled_Dataset(mapping_file, args)
        
parser = argparse.ArgumentParser(description='Deep feature extraction')
parser.add_argument('-db', required=True, help='''one of datasets: [controlled, uncontrolled, ...]''')
parser.add_argument('-db_config', required=True, type=int, help='integer: how to select and label images')
parser.add_argument('-patch_size', required=False, type=int, default=64, help='integer: one dimension of a patch')
parser.add_argument('-level', required=False, type=str, default='image', help='image level or patch level')

# --- for testing only --- #
if __name__ == '__main__':
    args = parser.parse_args()

    ds = get_datadist(args)

    if not os.path.isdir('dat'):
        os.makedirs('dat', exist_ok=True)

    ds.dump('dat/')
