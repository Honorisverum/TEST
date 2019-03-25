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
parser.add_argument("-epochs1", action="store", dest="epochs1", default=10, type=int)
parser.add_argument("-epochs2", action="store", dest="epochs2", default=10, type=int)
parser.add_argument("-seq_len", action="store", dest="seq_len", default=5, type=int)
parser.add_argument("-d_model", action="store", dest="d_model", default=64, type=int)
parser.add_argument("-lr", action="store", dest="lr", default=0.0001, type=float)
parser.add_argument("-A", action="store", dest="A", default=0.3, type=float)
parser.add_argument("-B", action="store", dest="B", default=0.1, type=float)
parser.add_argument("-I", action="store", dest="I", default=1.0, type=float)
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

cnn = cnn.PretrainedCNN(out_dim=args.d_model)
if args.transformer == "usual":
    transformer = transform.MakeTransformer(d_model=args.d_model,
                                            n_frames=args.seq_len+1,
                                            n_gts=args.seq_len, bb_dim=5)
elif args.transformer == "advanced":
    transformer = transform.MakeTransformerAdvanced(d_model=args.d_model,
                                                    n_frames=args.seq_len + 1,
                                                    n_gts=args.seq_len, bb_dim=5)
elif args.transformer == "lstm":
    transformer = transform.MakeLSTM(d_model=args.d_model,
                                     n_frames=args.seq_len + 1,
                                     n_gts=args.seq_len, bb_dim=5, use_gpu=use_gpu)

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
                          save_every=args.save_every, T=args.T, epochs1=args.epochs1,
                          epochs2=args.epochs2, use_gpu=use_gpu,
                          A=args.A, B=args.B, I=args.I, validating_set_videos=validating_set_videos)

