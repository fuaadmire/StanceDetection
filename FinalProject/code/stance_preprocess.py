import sys
import random
from utils.dataset import DataSet
from utils.generate_test_splits import get_body_ids, get_stances
from utils.score import LABELS

from utils.system import parse_params
import nltk
from tqdm import tqdm
import numpy as np
import os
import re
from six.moves.urllib.request import urlretrieve

reload(sys)
sys.setdefaultencoding('utf8')
random.seed(42)
np.random.seed(42)

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename.
    num_bytes=None disables the file size check."""
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print "Downloading file {}...".format(url + filename)
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix, filename), reporthook=reporthook(t))
        except AttributeError as e:
            print "An error occurred when downloading the file! Please get the dataset using a browser."
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print "File {} successfully loaded".format(filename)
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename



def get_tokenized_sequences(s):
    return [t for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s)).lower() #, flags=re.UNICODE

def write_to_file(out_file, line):
    out_file.write(line.encode('utf8') + '\n')

def preprocess_and_write(dataset, stance_list, tier, out_dir):
    #h, b, y = [],[],[]
    examples = []
    num_exmpls = 0
    for stance in tqdm(stance_list, desc="Preprocessing {}".format(tier)):
        y = LABELS.index(stance['Stance'])
        h = stance['Headline']
        b = dataset.articles[stance['Body ID']]
        
        h = clean(h)
        h_tokens = get_tokenized_sequences(h)
        
        b = clean(b)
        b_tokens = get_tokenized_sequences(b)
        examples.append((' '.join(h_tokens), ' '.join(b_tokens), ' '.join([str(y)])))
        num_exmpls = num_exmpls+1
    
    print("Processed %i examples" % (num_exmpls))
    
    # shuffle examples
    indices = range(len(examples))
    np.random.shuffle(indices)
    
    with open(os.path.join(out_dir, tier +'.headline'), 'w') as headline_file,  \
         open(os.path.join(out_dir, tier +'.body'), 'w') as body_file,\
         open(os.path.join(out_dir, tier +'.stance'), 'w') as stance_file:
             
        for i in indices:
            (headline, body, stance) = examples[i]
            # write tokenized data to file
            write_to_file(headline_file, headline)
            write_to_file(body_file, body)
            write_to_file(stance_file, stance)
    
        
    
    

if __name__ == "__main__":
    #check_version()
    args = parse_params()

    #Load the training dataset
    d = DataSet()
    #competition_dataset = DataSet(name="competition_test")
    
    
    train_body_ids, dev_body_ids = get_body_ids(d, training=0.8, base_dir='splits')
    train_stances = get_stances(d, train_body_ids)
    dev_stances = get_stances(d, dev_body_ids)
    
    #test_body_ids = list(competition_dataset.articles.keys())
    #test_stances = get_stances(competition_dataset, test_body_ids)

    
    print("Train data has %i examples total" % len(train_stances))
    print("Dev data has %i examples total" % len(dev_stances))
    #print("Test/Comp data has %i examples total" % len(test_stances))
    
    preprocess_and_write(d, train_stances, 'train', args.data_dir)
    preprocess_and_write(d, dev_stances, 'dev', args.data_dir)
    #preprocess_and_write(competition_dataset, test_stances, 'test', args.data_dir)
    