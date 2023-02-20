
import os
import numpy as np
import pandas as pd

import torch

import utils
from args import args
from model import *
from data_parse import parse_tdm1

#sets the device we train on as a GPU (cuda) if available or trains on local cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################
## Main runnings
#sets up paths for each fold of model (5-fold in total)
ckpt_path = os.path.join(args.save, f"fold_{args.fold}_model_{args.model}.ckpt")
eval_path = os.path.join(args.save, f"fold_{args.fold}_model_{args.model}.csv")
res_path = "rmse.csv"

########################################################################
#parses input data into feature columns, etc.
tdm1_obj = parse_tdm1(device, phase="test")
input_dim = tdm1_obj["input_dim"]
#represents hidden units of GRU in encoder
hidden_dim = 128 
latent_dim = 6

#instantiates encoder. Output dimension is 12 because 6 elements are used to determine the value of the mean for the distribution of the latent space while the other 6 are used to estimate the variance.
encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
#instantiates governing ODEFunc
ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=16)
#instantiates decoder
classifier = Classifier(latent_dim=latent_dim, output_dim=1)

#loads the model's parameter dictionary
utils.load_model(ckpt_path, encoder, ode_func, classifier, device)

########################################################################
## Predict & Evaluate
#disables gradient calculation, allowing for less memory consumption and faster compute. It is generally used to perform validation/testing because gradients are not required to be computed when testing model performance.
with torch.no_grad():
    #uses compute loss on test
    #the function compute_loss_ is where the ODE solver functions integrate the dosing info and time interval. This is also where the concept of VAE's are used (see compute_loss_on_test function) where z_0 is sampled from the latent distribution (which is derived from the mean and variance calculated by the 12 element input array). see page 6 of paper for more specific info. 
    test_res = utils.compute_loss_on_test(encoder, ode_func, classifier, args,
        tdm1_obj["test_dataloader"], tdm1_obj["n_test_batches"], 
        device, phase="test")

eval_results = pd.DataFrame(test_res).drop(columns="loss")
eval_results.to_csv(eval_path, index=False)

with torch.no_grad():
    #uses compute loss on interpolated data. Interpolated data contains estimated "intermediate" values between data points to smooth out the data
    test_res = utils.compute_loss_on_interp(encoder, ode_func, classifier, args,
        tdm1_obj["interp"], tdm1_obj["test_dataloader"], tdm1_obj["n_test_batches"], 
        device, phase="test")

#puts results in a data frame and migrates to csv file
eval_results = pd.DataFrame(test_res).drop(columns="loss")
eval_results.to_csv(eval_path + ".interp", index=False)

with torch.no_grad():
    #uses compute loss on interpolated data without dosing info
    test_res = utils.compute_loss_on_interp(encoder, ode_func, classifier, args,
        tdm1_obj["nodosing"], tdm1_obj["test_dataloader"], tdm1_obj["n_test_batches"], 
        device, phase="test")

eval_results = pd.DataFrame(test_res).drop(columns="loss")
eval_results.to_csv(eval_path + ".nodosing", index=False)



