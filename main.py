"""
=================================================
                    MAIN FILE
            FOR SETTING HYPER PARAMETERS
=================================================
"""


import loader
import torch
import argparse
import cnn
import training
from argparse import RawTextHelpFormatter
import transform

"""
=================================================
    HYPER PARAMETERS, CONSTANTS, NETWORK
=================================================
"""


parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)

parser.add_argument("-T", action="store", dest="T", default=5, type=int)
parser.add_argument("-img", action="store", dest="img", default=224, type=int)
parser.add_argument("-epochs", action="store", dest="epochs1", default=10, type=int)
parser.add_argument("-seq_len", action="store", dest="seq_len", default=5, type=int)
parser.add_argument("-d_model", action="store", dest="d_model", default=64, type=int)
parser.add_argument("-lr", action="store", dest="lr", default=0.0001, type=float)
parser.add_argument("-num_workers", action="store", dest="num_workers", default=0, type=int)
parser.add_argument("-save_every", action="store", dest="save_every", default=5, type=int)
parser.add_argument("-dir", action="store", dest="dir", default=".", type=str)
parser.add_argument("-transformer", action="store", dest="transformer", default="usual", type=str)


args = parser.parse_args()


for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")


"""
=================================================
    PREPARATION AND PREREQUISITES FOR RUN-UP
=================================================
"""

# load training/validation/test set
with open('./sets/train_set.txt') as f:
    training_set_titles = f.read().splitlines()
with open('./sets/valid_set.txt') as f:
    validating_set_titles = f.read().splitlines()

# GPU
use_gpu = torch.cuda.is_available()

cnn = cnn.PretrainedCNN(img_dim=args.img, out_dim=args.d_model)

transformer = transform.MakeTransformer(d_model=args.d_model,
                                        n_frames=args.seq_len+1,
                                        n_gts=args.seq_len, bb_dim=5)

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cnn = cnn.cuda()
    transformer = transformer.cuda()
    full_net = transform.MakeNet(cnn, transformer, args.seq_len).cuda()
    print('USE GPU', end="\n"*2)
else:
    full_net = transform.MakeNet(cnn, transformer, args.seq_len)
    print('USE CPU', end="\n"*2)
    

optimizer = torch.optim.Adam(full_net.parameters(),
                             lr=args.lr)


print("LOAD DATA VIDEOS...")
# prepare train, valid and test sets
training_set_videos = loader.load_videos(training_set_titles, use_gpu, "train set", args.num_workers, args.dir)
validating_set_videos = loader.load_videos(validating_set_titles, use_gpu, "valid set", args.num_workers, args.dir)
print("END LOADING!", end="\n"*2)



"""
=================================================
                TRAINING PHASE 
=================================================
"""


full_net = training.train(training_set_videos=training_set_videos,
                          net=full_net, optimizer=optimizer,
                          save_every=args.save_every, T=args.T, epochs=args.epochs1,
                          use_gpu=use_gpu, validating_set_videos=validating_set_videos)

