import os
from nn.models import Models
from bots import BotBid
from bidding import bidding
from util import hand_to_str
from deck52 import random_deal
from sample import Sample
import conf
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=200)

models = Models.from_conf(conf.load('ben/config/default.conf'),'..')   # loading neural networks
sampler = Sample.from_conf(conf.load('ben/config/default.conf'))  # Load sampling strategies

vuln_ns, vuln_ew = False, False

# you sit West and hold:
hand = '6.AKJT82.762.K63'

# the auction goes:
# (a few words about 'PAD_START':
# the auction is padded to dealer North
# if North is not dealer, than we have to put in a 'PAD_START' for every seat that was skipped
# if East deals we have one pad (because North is skipped)
# if South deals we have two pads (because North and East are skipped)
# etc.)
auction = ['1D', '3S']
bot_bid = BotBid([vuln_ns, vuln_ew], hand, models, sampler, 2, 0, False)

bid = bot_bid.bid(auction)
bid.to_dict()['candidates']

# what's your bid?
