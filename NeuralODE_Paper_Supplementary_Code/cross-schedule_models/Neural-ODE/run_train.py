
import os
import sys
import numpy as np
import pandas as pd
from random import SystemRandom
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

import utils
from model import *
from data_parse import parse_tdm1
from args import args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

input_cmd = sys.argv
input_cmd = " ".join(input_cmd)

########################################################################
## Main runnings
torch.manual_seed(args.random_seed + args.model + args.fold)
np.random.seed(args.random_seed + args.model + args.fold)

ckpt_path = os.path.join(args.save, f"fold_{args.fold}_model_{args.model}.ckpt")

########################################################################
tdm1_obj = parse_tdm1(device, phase="train")
input_dim = tdm1_obj["input_dim"]
hidden_dim = 128
latent_dim = 6

encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=16)
classifier = Classifier(latent_dim=latent_dim, output_dim=1)

if args.continue_train:
    utils.load_model(ckpt_path, encoder, ode_func, classifier, device)

########################################################################
## Train
log_path = "logs/" + f"fold_{args.fold}_model_{args.model}.log"
utils.makedirs("logs/")
logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
logger.info(input_cmd)

# batch size defines how many training samples must be done before updating weights/biases of a node during backprop
#epoch defines how many total backward passes we will do
batches_per_epoch = tdm1_obj["n_train_batches"]
#sets L2 norm-squared (MSE) between predicted and actual as loss criterion
criterion = nn.MSELoss().to(device=device)
params = (list(encoder.parameters()) + 
          list(ode_func.parameters()) + 
          list(classifier.parameters()))
#utilizing Adam optimization algorithm rather than SGD to overcome saddlepoints in data. It incorporates the idea of momentum by nudging weights/biases by the average running gradient rather than the gradient itself.
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.l2)
best_rmse = 0x7fffffff
best_epochs = 0

for epoch in range(1, args.epochs):

    for _ in tqdm(range(batches_per_epoch), ascii=True):
        #sets gradients of all parameters to zero. This prevents the incorrect accumulation of gradients that occurs if you call loss.backwards more than once wihtout zeroing out the gradients.
        optimizer.zero_grad()

        #extracts training features and dosing
        ptnms, times, features, labels, cmax_time = tdm1_obj["train_dataloader"].__next__()
        dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
        dosing[:, :, 0] = features[:, :, -2]
        dosing = dosing.permute(1, 0, 2)

        #VAE concept used here. We are taking the output of the encoder and sampling z_0 from a distribution of the latent space variables gathered from estimating the mean and variance from the 12 elements outputted by the encoder. 
        encoder_out = encoder(features)
        qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
        z0 = utils.sample_standard_gaussian(qz0_mean, qz0_var)
        
        solves = z0.unsqueeze(0).clone()
        try:
            #this is where the idea of neural-ODE's are used. dosing information and time interval are incorporated into the event from the previous time step. The time interval from the previous time interval and z_i-1 are sent into the ODE Solver function.
            for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                z0 += dosing[idx]
                time_interval = torch.Tensor([time0 - time0, time1 - time0])
                #ODE Solver function 
                sol = odeint(ode_func, z0, time_interval, rtol=args.tol, atol=args.tol)
                z0 = sol[-1].clone()
                #running sequence of all z_i's at each time step which will eventially be used to predict output in the decoder
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(times)
            print(time0, time1, time_interval, ptnms)
            continue

        #prediction generation from sequence of z_i's
        preds = classifier(solves, cmax_time)

        # computes MSE on preds vs observations 
        loss = utils.compute_loss_on_train(criterion, labels, preds)
        try: 
            #automatically computes gradients of loss tensor
            loss.backward()
        except RuntimeError:
            print(ptnms)
            print(times)
            continue
        #performs a single parameter update (single optimization step)
        optimizer.step()
    
    idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
    preds = preds.permute(1, 0, 2)[idx_not_nan]
    labels = labels[idx_not_nan]
    print(preds)
    print(labels)

    #torch.no_grad() is used to prevent the automatic calculation of gradients to clearly see unbiased training/validation error 
    with torch.no_grad():
        
        #training error
        train_res = utils.compute_loss_on_test(encoder, ode_func, classifier, args,
            tdm1_obj["train_dataloader"], tdm1_obj["n_train_batches"], 
            device, phase="train")

        #validation error
        validation_res = utils.compute_loss_on_test(encoder, ode_func, classifier, args,
            tdm1_obj["val_dataloader"], tdm1_obj["n_val_batches"], 
            device, phase="validate")
        
        train_loss = train_res["loss"] 
        validation_loss = validation_res["loss"]

        #if the validation loss on the current interation is better than the best running RMSE, then we save the weights and biases of the encoder, ode, classifier, and arguments
        if validation_loss < best_rmse:
            torch.save({'encoder': encoder.state_dict(),
                        'ode': ode_func.state_dict(),
                        'classifier': classifier.state_dict(),
                        'args': args}, ckpt_path)
            best_rmse = validation_loss
            best_epochs = epoch

        message = """
        Epoch {:04d} | Training loss {:.6f} | Training R2 {:.6f} | Validation loss {:.6f} | Validation R2 {:.6f}
        Best loss {:.6f} | Best epoch {:04d}
        """.format(epoch, train_loss, train_res["r2"], validation_loss, validation_res["r2"], best_rmse, best_epochs)
        logger.info(message)
        
        


