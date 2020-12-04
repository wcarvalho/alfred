import sys
import os
import json
import collections
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def main(data, in_splits, out_splits, train_scenes=[], test_scenes=[]):
  train_scenes = set(train_scenes)
  test_scenes = set(test_scenes)
  with open(in_splits) as f:
    splits = json.load(f)

  new_splits = {k:[] for k in splits.keys()}
  for k, task_dicts in splits.items():

    for task_dict in tqdm(task_dicts):
      json_path = os.path.join(data, k, task_dict['task'], 'traj_data.json')

      with open(json_path) as f:
        ex = json.load(f)

      scene_number = ex['scene']['scene_num']
      if 'train' in k:
        if scene_number in train_scenes:
          new_splits[k].append(task_dict)
      else:
        if scene_number in test_scenes:
          new_splits[k].append(task_dict)

  original_sizes = {k:len(d) for k, d in splits.items()}
  new_sizes = {k:len(d) for k, d in new_splits.items()}
  ratios = {k: "%.2f" % (100*float(new_sizes[k])/float(original_sizes[k])) for k in splits.keys()}

  print("="*10, "Reduction (%)", "="*10)
  print(ratios)

  with open(out_splits, 'w') as outfile:
    json.dump(new_splits, outfile)
    print(f'Write {out_splits}')





if __name__ == '__main__':
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
  parser.add_argument('--in-splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
  parser.add_argument('--out-splits', help='json file containing train/dev/test splits', default='data/splits/kitchens.json')
  parser.add_argument('--train-scenes', type=int, nargs="+", help='scenes to allow for training')
  parser.add_argument('--test-scenes', type=int, nargs="+", help='scenes to allow for testing')
  args = parser.parse_args()
  main(**vars(args))