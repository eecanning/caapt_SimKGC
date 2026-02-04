import os
import glob

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None


def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
        global all_triplet_dict
        if not all_triplet_dict:
            data_dir = os.path.dirname(args.train_path)

            # Only consider known triplet filenames (train/valid/test and their _aug variants).
            # This avoids accidentally loading non-triplet JSONs such as entities.json,
            # entity2id.json, or relation2id.json which will break TripletDict.
            candidates = [
                'train.json', 'train_aug.json',
                'valid.json', 'valid_aug.json',
                'test.json', 'test_aug.json'
            ]

            path_list = []
            for fn in candidates:
                p = os.path.join(data_dir, fn)
                if os.path.exists(p):
                    path_list.append(p)

            if not path_list:
                # fallback: if none of the standard candidates exist, try the previous pattern
                # but prefer to warn so users can correct their data layout.
                logger.warning('No standard triplet files found in {}. Falling back to globbing *.json.'.format(data_dir))
                path_list = glob.glob(os.path.join(data_dir, '*.json'))

            all_triplet_dict = TripletDict(path_list=path_list)


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
