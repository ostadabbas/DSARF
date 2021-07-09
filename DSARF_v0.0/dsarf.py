
import numpy as np
import argparse
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable           
import os
import time
import sys
'Unloading matplotlib to load it later with Agg backend'
modules = []
for module in sys.modules:
    if module.startswith('matplotlib'):
        modules.append(module)
for module in modules:
    sys.modules.pop(module)
'##############'
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))
import pdb
from tqdm import tqdm


def FC(shape = None, init = None):
    if init is None:
        K = shape[-2]
        init = [torch.rand(shape) * 2 - 1]
        shape_bias = shape.copy()
        shape_bias[-2] = 1
        init.append(torch.rand(shape_bias) * 2 - 1)
    else:
        K = init[0].shape[-2]
    fc = nn.Parameter(init[0] * np.sqrt(1/K))
    fc_bias = nn.Parameter(init[1] * np.sqrt(1/K))
    return fc, fc_bias

class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1}, s_t)
    """
    def __init__(self, z_dim, u_dim, transition_dim, S, L):
        super(GatedTransition, self).__init__()
        # initialize the linear transformations used in the neural network
        # g (scalar)
        self.fc1_g, self.fc1_g_bias = FC([S, L, z_dim + u_dim, transition_dim])
        self.fc2_g, self.fc2_g_bias = FC([S, transition_dim, z_dim])
        # nonlinear z transition
        self.fc1_z, self.fc1_z_bias = FC([S, L, z_dim + u_dim, transition_dim])
        self.fc2_z, self.fc2_z_bias = FC([S, transition_dim, z_dim])
        self.fc3_z, self.fc3_z_bias = FC([S, z_dim, z_dim])
        # linear z transition
        init = [torch.eye(z_dim + u_dim, z_dim).repeat(S, L, 1, 1),
                torch.zeros(1, z_dim).repeat(S, L, 1, 1)]
        self.fc_z, self.fc_z_bias = FC(init = init)
        # initialize the non-linearities used in the neural network
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t_1, u_t_1):
        """
        Given the latent z_{t-1} and stimuli u_{t-1} corresponding to the time
        step t-1, we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1}, u_{t-1})
        z,u_{t-1} is L * Batch * z_dim
        """
        # stack z and u in a single vector if u is available
        if u_t_1 is not None:
            zu_t_1 = torch.cat((z_t_1, u_t_1), dim = -1)
        else:
            zu_t_1 = z_t_1
        _gate = self.relu(torch.matmul(zu_t_1, self.fc1_g) + self.fc1_g_bias) 
        gate = self.sigmoid(torch.matmul(_gate.mean(dim = 1), self.fc2_g) + self.fc2_g_bias)
        # compute the 'proposed mean'
        _z_mean = self.relu(torch.matmul(zu_t_1, self.fc1_z) + self.fc1_z_bias) 
        z_mean = torch.matmul(_z_mean.mean(dim = 1), self.fc2_z) + self.fc2_z_bias
        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        z_mean_lin = torch.matmul(zu_t_1, self.fc_z) + self.fc_z_bias
        z_loc = (1 - gate) * z_mean_lin.mean(dim = 1)  + gate * z_mean
        # compute the scale used to sample z_t, using the proposed
        # mean from above as input
        z_scale = torch.matmul(self.relu(z_mean), self.fc3_z) + self.fc3_z_bias
        # return loc, scale which can be fed into Normal: S * Batch * z_dim
        return z_loc, z_scale
    

class StateTransition(nn.Module):
    """
    Parameterizes the categorical latent transition probability p(s_t |s_{t-1})
    """
    def __init__(self, S):
        super(StateTransition, self).__init__()
        # linear s transition
        self.fc_s = nn.Linear(S, S)
        # initialize the activation used in the transition
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, s_t_1):
        """
        Given the latent s_{t-1}, we return the probabilities
        that parameterize the cateorical distribution p(s_t | s_{t-1})
        """
        s_t = self.softmax(self.fc_s(s_t_1))
        return s_t



class SpatialFactors(nn.Module):
    """
    Parameterizes spatial factors  p(F | z_F)
    """
    def __init__(self, D, factor_dim, zF_dim):
        super(SpatialFactors, self).__init__()
        # initialize the linear transformations used in the neural network
        # shared structure
        self.fc1 = nn.Linear(zF_dim, 2*zF_dim)
        self.fc2 = nn.Linear(2*zF_dim, 4*zF_dim)
        # mean and sigma for factor location
        self.fc3 = nn.Linear(4*zF_dim, 2 * D * factor_dim)
        # initialize the non-linearities used in the neural network
        self.relu = nn.PReLU()
        
    def forward(self, z_F):
        """
        Given the latent z_F corresponding to spatial factor embedding
        we return the mean and sigma vectors that parameterize the
        (diagonal) gaussian distribution p(F | z_F) for factor location 
        and scale.
        """
        # computations for shared structure 
        _h = self.relu(self.fc1(z_F))
        h = self.relu(self.fc2(_h))
        # compute the 'mean' and 'sigma' for factor location given z_F
        factor_params = self.fc3(h)
        # return means, sigmas of factor loc, scale which can be fed into Normal
        return factor_params[:,:D*factor_dim], factor_params[:,D*factor_dim:]
    

class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, w_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on w_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """
    def __init__(self, z_dim, u_dim, rnn_dim, L):
        super(Combiner, self).__init__()
        # initialize the linear transformations used in the neural network
        self.fc1_z, self.fc1_z_bias = FC([L, z_dim + u_dim, rnn_dim])
        self.fc2_z = nn.Linear(rnn_dim, z_dim)
        self.fc3_z = nn.Linear(rnn_dim, z_dim)
        # initialize the non-linearities used in the neural network
        self.tanh = nn.Tanh()

    def forward(self, z_t_1, u_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(w_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, y_{t:T})
        """
        # stack z and u in a single vector if u is available
        if u_t_1 is not None:
            zu_t_1 = torch.cat((z_t_1, u_t_1), dim = -1)
        else:
            zu_t_1 = z_t_1
        # combine the rnn hidden state with a transformed version of z_t_1
        h = torch.matmul(zu_t_1, self.fc1_z) + self.fc1_z_bias
        h_combined = 0.5 * (self.tanh(h).mean(dim = 0) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.fc2_z(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.fc3_z(h_combined)
        # return loc, scale which can be fed into Normal
        return loc, scale


class DSARF(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution for the Deep Markov Factor Analysis
    """
    def __init__(self, n_data=100, T = 18,  factor_dim=10, z_dim=5,
                 u_dim=0, transition_dim=5, zF_dim=5, n_class=1,
                 sigma_obs = 0, rnn_dim = None, rnn_dropout_rate = 0.0,
                 D = 300, L = None, S = None):
        super(DSARF, self).__init__()
        
        self.rnn_dim = rnn_dim
        # observation noise
        self.sig_obs = sigma_obs
        self.L = L
        self.S = S
        # instantiate pytorch modules used in the model and guide below
        self.trans = GatedTransition(z_dim, u_dim, transition_dim, S, len(L))
        self.strans = StateTransition(S)
        self.spat = SpatialFactors(D, factor_dim, zF_dim)
        
        if rnn_dim is not None: # initialize extended DSARF
            self.combiner = Combiner(z_dim, u_dim, rnn_dim*2, len(L))
            self.rnn = nn.RNN(input_size=factor_dim, hidden_size=rnn_dim,
                              nonlinearity='relu', batch_first=True,
                              bidirectional=True, num_layers=1, dropout=rnn_dropout_rate)
            # define a (trainable) parameter for the initial hidden state of the rnn
            self.h_0 = nn.Parameter(torch.zeros(2, 1, rnn_dim))

        self.softmax = nn.Softmax(dim = -1)
        self.p_c = nn.Parameter(torch.ones(1, n_class))
        self.p_s_0 = nn.Parameter(torch.ones(1, S))
        
        self.p_z_F_mu = nn.Parameter(torch.zeros(1, zF_dim))
        self.p_z_F_sig = nn.Parameter(torch.ones(1, zF_dim).log())

        self.z_0_mu = nn.Parameter(torch.rand(max(L), n_class, z_dim)- 1/2)
        init_sig = (torch.ones(max(L), n_class, z_dim) / (2 * n_class)* 0.15 * 5).log()
        self.z_0_sig = nn.Parameter(init_sig)
        
        self.q_c = torch.ones(n_data, max(L), n_class) / n_class
        if args.ID == 'apnea':
            self.q_s = nn.Parameter(torch.ones(n_data, T, S) / S, requires_grad=False)
        else:
            self.q_s = torch.ones(n_data, T, S) / S
        self.q_s_0 = nn.Parameter(torch.ones(n_data, S))
        
        self.q_z_0_mu = nn.Parameter(torch.rand(n_data, max(L), z_dim)- 1/2)
        init_sig = (torch.ones(n_data, max(L), z_dim) / (2 * n_class)*0.1).log()
        self.q_z_0_sig = nn.Parameter(init_sig)
        if rnn_dim is not None:
            self.q_z_mu = torch.zeros(n_data, T, z_dim)
            self.q_z_sig = torch.ones(n_data, T, z_dim)
        else:
            self.q_z_mu = nn.Parameter(torch.rand(n_data, T, z_dim)- 1/2)
            init_sig = (torch.ones(n_data, T, z_dim) / (2 * n_class) * 0.1).log()
            self.q_z_sig = nn.Parameter(init_sig)
        self.q_z_F_mu = nn.Parameter(torch.zeros(1, zF_dim))
        init_sig = torch.ones(1, zF_dim).log()
        self.q_z_F_sig = nn.Parameter(init_sig)
        self.q_F_loc_mu = nn.Parameter(torch.rand(factor_dim, D)- 1/2)
        init_sig = (torch.ones(factor_dim, D) / (2 * factor_dim) * 0.1).log()
        self.q_F_loc_sig = nn.Parameter(init_sig)
                  
    def Reparam(self, mu_latent, sigma_latent):
        eps = Variable(mu_latent.data.new(mu_latent.size()).normal_())
        return eps.mul(sigma_latent.exp()).add_(mu_latent)
    
    # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
    def forward(self, mini_batch, u_values, mini_batch_idxs):
        # u_values = (data_points, time_points, u_dim)
        # z_values = (data_points, time_points + max(L), z_dim)
        # F_loc_values = (factor_dim, D)
        N = mini_batch.size(0)
        T_b = mini_batch.size(1)
        z_dim = self.q_z_0_mu.size(-1)
        n_class = self.z_0_mu.size(1)
        
        q_z_0_mus = self.q_z_0_mu[mini_batch_idxs] #batch*L*z_dim
        q_z_0_sigs = self.q_z_0_sig[mini_batch_idxs] #batch*L*z_dim
        z_0_values = self.Reparam(q_z_0_mus, q_z_0_sigs)
        
        if self.rnn_dim is not None:
            q_z_mus = torch.Tensor([]).reshape(N, 0, z_dim)
            q_z_sigs = torch.Tensor([]).reshape(N, 0, z_dim)
            h_0_contig = self.h_0.expand(2, T_b,
                                         self.rnn.hidden_size).contiguous()
            rnn_output, _= self.rnn(mini_batch.permute(1,0,2), h_0_contig)
            z_values = z_0_values.clone()
            z_prev = z_values.permute(1,0,2)[-np.array(self.L)]
            for i in range(T_b):
                if u_values is not None and max(self.L) == 1:
                    u_prev = u_values[:,i,:].unsqueeze(0)
                else:
                    u_prev = None
                loc, scale = self.combiner(z_prev, u_prev, rnn_output[i])
                z_val = self.Reparam(loc, scale)
                z_values = torch.cat((z_values,z_val.unsqueeze(1)), dim=1)
                z_prev = z_values.permute(1,0,2)[-np.array(self.L)]
                q_z_mus = torch.cat((q_z_mus,loc.unsqueeze(1)), dim=1)
                q_z_sigs = torch.cat((q_z_sigs,scale.unsqueeze(1)), dim=1)            
            self.q_z_mu[mini_batch_idxs, :T_b] = q_z_mus
            self.q_z_sig[mini_batch_idxs, :T_b] = q_z_sigs
        else:
            q_z_mus = self.q_z_mu[mini_batch_idxs, :T_b] #batch*T*z_dim
            q_z_sigs = self.q_z_sig[mini_batch_idxs, :T_b] #batch*T*z_dim
            z_t_values = self.Reparam(q_z_mus, q_z_sigs)
            z_values = torch.cat((z_0_values, z_t_values), dim = 1)   
              
        # p(z_t|z_{t-1},u{t-1}, s_t) = Normal(z_loc, z_scale)
        z_t_1 = torch.Tensor([]).reshape(0, N * T_b, z_dim)
        for lag in self.L:
            z_t_1 = torch.cat((z_t_1,
                               z_values[:, max(self.L)-lag:-lag].reshape(1, -1, z_dim)),
                              dim = 0)
        if u_values is not None and max(self.L) == 1:
            u_t_1 = u_values.reshape(1, N * T_b, -1)
        else:
            u_t_1 = None
        p_z_mu, p_z_sig = self.trans(z_t_1, u_t_1)
        p_z_mu = p_z_mu.view(self.S, N, T_b, -1)
        p_z_sig = p_z_sig.view(self.S, N, T_b, -1)
        
        # compute q(s_0)
        p_s_0 = self.softmax(self.p_s_0)
        q_s_0 = self.softmax(self.q_s_0[mini_batch_idxs])
        # compute q(s_t) = p(s_t|z_t) = p(z_t|z_{t-1},s_t)p(s_t|s_{t-1})
        q_s_t = self.q_s[mini_batch_idxs, :T_b]
        q_s = torch.cat((q_s_0.unsqueeze(1), q_s_t), dim=1)
        p_s_t = self.strans(q_s[:,:-1].reshape(N*T_b, -1)).reshape(N, T_b, -1)
        z_t_vals = z_values[:, max(self.L):] # batch*T_b*z_dim
        # compute q(s_t)
        q_s_t = p_s_t.permute(2, 0, 1).log()\
              -1/2*((z_t_vals - p_z_mu)\
              /(p_z_sig.exp()+1e-4)).pow(2).sum(dim = -1)\
              -p_z_sig.sum(dim = -1)
        q_s_t = self.softmax(q_s_t.permute(1, 2, 0))
        self.q_s[mini_batch_idxs, :T_b] = q_s_t.detach()
        if self.S == 1:
            q_s_t = torch.ones(N, T_b, 1)
        
        z_0_vals = z_0_values.repeat(1, 1, n_class).view(N, -1, n_class, z_dim)
        q_c = self.softmax(self.p_c).log()\
              -1/2*((z_0_vals - self.z_0_mu)\
              /(self.z_0_sig.exp()+1e-4)).pow(2).sum(dim = -1)\
              -self.z_0_sig.sum(dim = -1)
        q_c = self.softmax(q_c)
        self.q_c[mini_batch_idxs] = q_c.detach()

        F_loc_values = self.Reparam(self.q_F_loc_mu,
                                     self.q_F_loc_sig)
        zF_value = self.Reparam(self.q_z_F_mu,
                                 self.q_z_F_sig)
        
        # p(F_mu|z_F) = Normal(F_mu_loc, F_mu_scale)
        p_F_loc_mu, p_F_loc_sig = self.spat(zF_value)
        p_F_loc_mu = p_F_loc_mu.view(factor_dim, -1)
        p_F_loc_sig = p_F_loc_sig.view(factor_dim, -1)
        
        # p(y|z,F) = Normal(z*F, sigma)
        # z : (data_points, time_points, factor_dim)
        # F: (data_points, factor_dim, voxel_num)
        y_hat_nn = torch.matmul(z_values[:, max(self.L):],
                                F_loc_values)
        obs_noise = y_hat_nn.data.new(y_hat_nn.size()).normal_()
        y_hat = obs_noise.mul(self.sig_obs).add_(y_hat_nn)
        
        q_z_0_mus.unsqueeze_(2)
        q_z_0_sigs.unsqueeze_(2)
        
        return y_hat,\
                q_c, self.p_c,\
                q_s_0, p_s_0,\
                q_s_t, p_s_t,\
                q_z_0_mus, q_z_0_sigs,\
                self.z_0_mu, self.z_0_sig,\
                q_z_mus, q_z_sigs,\
                p_z_mu, p_z_sig,\
                self.q_F_loc_mu, self.q_F_loc_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                self.q_z_F_mu, self.q_z_F_sig,\
                self.p_z_F_mu, self.p_z_F_sig
#batch*L*n_class, 1*n_class               
#batch*S, 1*S
# batch*T*S
# batch*L*1*z_dim
# L*n_class*z_dim
# batch*T*z_dim
# S*batch*T*z_dim
#factor*D
# factor*D
# 1*zF_dim
# 1*zF_dim
def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    KLD = 1/2 * ( 2 * (p_sigma - q_sigma) 
                    - 1
                    + ((q_sigma.exp())/(p_sigma.exp()+1e-6)).pow(2)
                    + ( (p_mu - q_mu) / (p_sigma.exp()+1e-6) ).pow(2) )
    return KLD.sum(dim = -1)

def KLD_Cat(q, p):
    # sum (q log (q/p) )
    KLD = q * ((q+1e-4) / (p+1e-4)).log()
    return KLD.sum(dim = -1)

mse_loss = torch.nn.MSELoss(size_average=False, reduce=True)

def ELBO_Loss(mini_batch, y_hat,\
              q_c, p_c,\
              q_s_0, p_s_0,\
              q_s_t, p_s_t,\
              q_z_0_mus, q_z_0_sigs,\
              z_0_mu, z_0_sig,\
              q_z_mus, q_z_sigs,\
              p_z_mu, p_z_sig,\
              q_F_loc_mu, q_F_loc_sig,\
              p_F_loc_mu, p_F_loc_sig,\
              q_z_F_mu, q_z_F_sig,\
              p_z_F_mu, p_z_F_sig,\
              annealing_factor = 1):
    
    rec_loss = mse_loss(y_hat, mini_batch)
    KL_s_0 = KLD_Cat(q_s_0.mean(dim=0), p_s_0).sum()
    KL_s_t = KLD_Cat(q_s_t, p_s_t).sum()
    if S == 1:
        KL_s_t = 0
    KL_z_0 = (q_c *
                KLD_Gaussian(q_z_0_mus, q_z_0_sigs,
                             z_0_mu, z_0_sig)).sum()
    KL_z = (q_s_t.permute(2,0,1) *
                  KLD_Gaussian(q_z_mus, q_z_sigs, 
                               p_z_mu, p_z_sig)).sum()
    KL_F_loc = KLD_Gaussian(q_F_loc_mu, q_F_loc_sig,
                            p_F_loc_mu, p_F_loc_sig).sum()
    KL_z_F = KLD_Gaussian(q_z_F_mu, q_z_F_sig,
                          p_z_F_mu, p_z_F_sig).sum()

    return rec_loss + annealing_factor * (KL_s_0 + KL_s_t
                                          + KL_z_0 + KL_z
                                          + KL_F_loc + KL_z_F)


# parse command-line arguments and execute the main method
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-k', '--factor_dim', type=int, default=100)
    parser.add_argument('-dt', '--transition_dim', type=int, default=100)
    parser.add_argument('-du', '--u_dim', type=int, default=0)
    parser.add_argument('-dzf', '--zF_dim', type=int, default=2)
    parser.add_argument('-c', '--n_class', type=int, default=1)
    parser.add_argument('-s', '--n_state', type=int, default=1)
    parser.add_argument('-lag', nargs='+', type=int, default=1)
    parser.add_argument('-so', '--sigma_obs', type=float, default=0)
    parser.add_argument('-restore', action="store_true", default=False)
    parser.add_argument('-resume', action="store_true", default=False)
    parser.add_argument('-predict', action="store_true", default=False)
    parser.add_argument('-strain', action="store_false", default=True,
                        help = "whether to save train results every 50 epochs")
    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('-last', type=int, default=1)
    parser.add_argument('-long', action="store_true", default=False)
    parser.add_argument('-ID', type=str, default=None)
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-drnn', '--rnn_dim', type=int, default=None)
    parser.add_argument('-file', '--data_file', type=str, default='./data_traffic/tensor.mat')
    parser.add_argument('-smod', '--model_path', type=str, default='./ckpt_files/')
    parser.add_argument('-dpath', '--dump_path', type=str, default='./')
    args = parser.parse_args()

    'Code starts Here'
    # we have N number of data points each of size (T*D)
    # we have N number of u each of size (T*u_dim)
    # we specify a vector with values 0, 1, ..., N-1
    # each datapoint for shuffling is a tuple ((T*D), (T*u_dim), n)
    
    if args.long:
        if args.ID == 'pacific':
            from plot_results_pacific_long_term import plot_result
        else:
            from plot_results_long_term import plot_result
    else:
        if args.ID == 'apnea':
            from plot_results_apnea import plot_result  
        elif args.ID == 'pacific':
            from plot_results_pacific_short_term import plot_result 
        else:
            from plot_results_short_term import plot_result
    # setting hyperparametrs
    
    factor_dim = args.factor_dim # number of Gaussian blobs (spatial factors)
    u_dim = args.u_dim # dimension of stimuli embedding
    zF_dim = args.zF_dim # dimension of spatial factor embedding
    n_class = args.n_class # number of major clusters
    S = args.n_state
    sigma_obs = args.sigma_obs # standard deviation of observation noise
    T_A = 100 # annealing iterations <= epoch_num
    Restore = args.restore # set to True if already trained
    predict = args.predict
    load_model = args.resume
    batch_size = args.batch_size # batch size
    epoch_num = args.epoch # number of epochs
    num_workers = 0 # number of workers to process dataset
    lr = args.lr # learning rate for adam optimizer
    z_dim = args.factor_dim # dimension of temporal latent variable z
    if args.ID == 'flu' and args.long:
        transition_dim = 2
    else:
        transition_dim = args.factor_dim # hidden units from z_{t-1} to z_t
    if args.rnn_dim is not None: # whether to add rnn extension to files/folders
        rnn_ext = '_rnn'
    else:
        rnn_ext = ''
    # dataset parameters
    root_dir = args.data_file
    # Path parameters
    save_PATH = args.model_path
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
    fig_PATH = args.dump_path + 'fig_files%s/' %(rnn_ext)
    if not os.path.exists(fig_PATH):
        os.makedirs(fig_PATH)
    PATH_DSARF = save_PATH + 'DSARF%s' %(rnn_ext)
    
    """
    DSARF SETUP & Training
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    """
    spat = None
    if root_dir[-3:] == 'son':
        import json
        f = open(root_dir)
        data_st = json.load(f)
        if args.ID == 'bat':
            D = len(data_st['joints']) * 3
            spat = data_st['joints']
            data_st = [np.array(data_st[key]).reshape(-1, D)
                       for key in list(data_st.keys())[3:-2]]
            if args.long:
                tpred = sum([len(data_st[-i]) for i in range(args.last,0,-1)]) 
                data_st = np.concatenate(data_st, axis = 0).reshape(1,-1,D)
        else:
            data_st = [np.array(i) for i in data_st]
            D = data_st[0].shape[-1]
            if args.long:
                tpred = sum([len(data_st[-i]) for i in range(args.last,0,-1)])
                data_st = np.concatenate(data_st, axis = 0).reshape(-1,1,D)
            

    if root_dir[-3:] == 'txt':
        import pandas as pd
        if args.ID in ['flu', 'dengue']:
            data_st = pd.read_csv(root_dir, sep=',')
            years = np.array([i[:4] for i in data_st.values[:,0]], dtype = np.int)
            data_st = [data_st.values[years==i,1:].astype('float') for i in range(2003, 2016)]
            D = data_st[0].shape[-1]
            if args.long:
                tpred = sum([len(data_st[-i]) for i in range(args.last,0,-1)])
                data_st = np.concatenate(data_st, axis = 0).reshape(1,-1,D)

    if root_dir[-3:] == 'mat':
        from scipy.io import loadmat
        data_st = loadmat(root_dir)['tensor'].transpose(1,2,0).astype('float')  
    if root_dir[-3:] == 'npz':
        data_st = np.load(root_dir)['arr_0'].reshape(-1,28,288).transpose(1,2,0).astype('float')

    if args.ID == 'pacific':
        import pandas as pd
        data_st = pd.read_csv(root_dir, sep='\t', header = None).values.reshape(-1,30*84)[3:].reshape(-1,12,30*84)
        ## 2520 spatal locations in a grid of 30*84 for 399 months from Jan 1970 to March 2003

    if args.ID == 'apnea':        
        import pandas as pd
        data_st = [pd.read_csv(root_dir, sep=' ', header = None).values[6201:7201,1].reshape(-1,1),
                   pd.read_csv(root_dir, sep=' ', header = None).values[5201:6201,1].reshape(-1,1)]
        
        states = None
        D = data_st[0].shape[-1]
        if args.long:
            tpred = sum([len(data_st[-i]) for i in range(args.last,0,-1)])
            data_st = data_st.reshape(1,-1,D)

 
    if args.ID in ['guangzhou','birmingham','hangzhou','seattle', 'pacific']:
        D = data_st.shape[-1]
        data_st[data_st == 0] = np.nan
        if args.long:
            tpred = sum([len(data_st[-i]) for i in range(args.last,0,-1)])
            data_st = data_st.reshape(1,-1,D)
        data_st_true = data_st.copy()
        if args.long:
            data_st[:,-tpred:] = np.nan
        data_mean = [data_st[~np.isnan(data_st)].mean() for i in data_st]
        data_std = [data_st[~np.isnan(data_st)].std() for i in data_st]
    elif args.ID in ['flu','dengue','prec','apnea']:
        data_st_cat = np.concatenate(data_st, axis = 0)
        data_mean = [data_st_cat[~np.isnan(data_st_cat)].mean() for i in data_st]
        data_std = [data_st_cat[~np.isnan(data_st_cat)].std() for i in data_st]
        data_st_true = data_st.copy()
        if args.long:
            data_st[:,-tpred:] = np.nan
    else:
        if args.long:
            tpred = sum([len(data_st[-i]) for i in range(args.last,0,-1)])
            data_st = data_st.reshape(1,-1,D)
        data_st_true = data_st.copy()
        if args.long:
            data_st[:,-tpred:] = np.nan
        data_mean = [i[~np.isnan(i)].mean() for i in data_st]
        data_std = [i[~np.isnan(i)].std() for i in data_st]
    # normalize data for training
    dataa = [(data_st_true[i] - data_mean[i])/data_std[i] for i in range(len(data_st_true))]
    dataa_train = [(data_st[i] - data_mean[i])/data_std[i] for i in range(len(data_st))]
    
    # set parameters 
    n_data = len(dataa_train)
    T = max([len(i) for i in dataa_train]) # use maximum T to conveniently support varying length
    #form data for training
    training_set = [(torch.FloatTensor(y),torch.zeros(0),torch.LongTensor([i])) for i, y in enumerate(dataa_train)]
    # exclude test-set from training 
    if predict:
        training_set_part =  training_set[-args.last:]
        load_model = True
    else:
        if args.long:
            training_set_part =  training_set.copy()          
        else:
            training_set_part =  training_set[:-args.last]
    if args.long:
        t = tpred
    else:
        t=args.last
    # form classes--It's dummy for these set of experiments
    classes = torch.LongTensor(np.zeros(n_data))
    # initialize model
    dsarf = DSARF(n_data = n_data,
                T = T,
                factor_dim = factor_dim,
                z_dim = z_dim,
                u_dim = u_dim,
                transition_dim = transition_dim,
                zF_dim = zF_dim,
                n_class = n_class,
                sigma_obs = sigma_obs,
                rnn_dim = args.rnn_dim,
                D = D,
                L = args.lag,
                S = args.n_state)
    

    # freeze model for prediction
    if predict:
        
        dsarf.p_c.requires_grad = False
        dsarf.p_s_0.requires_grad = False
        dsarf.z_0_mu.requires_grad = False
        dsarf.z_0_sig.requires_grad = False
        
        dsarf.q_F_loc_mu.requires_grad = False
        dsarf.q_F_loc_sig.requires_grad = False
        dsarf.q_z_F_mu.requires_grad = False
        dsarf.q_z_F_sig.requires_grad = False
        for param in dsarf.trans.parameters():
            param.requires_grad = False
        for param in dsarf.spat.parameters():
            param.requires_grad = False
        for param in dsarf.strans.parameters():
            param.requires_grad = False
            
    if Restore == False:
        # set path to save figure results during training
        fig_PATH_train = fig_PATH + 'figs_train/'
        if not os.path.exists(fig_PATH_train):
            os.makedirs(fig_PATH_train)
        
        optim_dsarf = optim.Adam(dsarf.parameters(), lr = lr)
        # number of parameters  
        total_params = sum(p.numel() for p in dsarf.parameters())
        learnable_params = sum(p.numel() for p in dsarf.parameters() if p.requires_grad)
        print('Total Number of Parameters: %d' % total_params)
        print('Learnable Parameters: %d' %learnable_params)
        
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'num_workers': num_workers}
        train_loader = data.DataLoader(training_set_part, **params)
        
        print("Training...")
        if load_model:
            dsarf.load_state_dict(torch.load(PATH_DSARF,
                              map_location=lambda storage, loc: storage))
        for i in range(epoch_num):
            time_start = time.time()
            loss_value = 0.0
            for batch_indx, batch_data in enumerate(tqdm(train_loader)):
            # update DSARF
                mini_batch, u_vals, mini_batch_idxs = batch_data
                mini_batch_idxs = mini_batch_idxs.reshape(-1)
                if u_dim == 0:
                    u_vals = None
    
                y_hat,\
                q_c, p_c,\
                q_s_0, p_s_0,\
                q_s_t, p_s_t,\
                q_z_0_mus, q_z_0_sigs,\
                z_0_mu, z_0_sig,\
                q_z_mus, q_z_sigs,\
                p_z_mu, p_z_sig,\
                q_F_loc_mu, q_F_loc_sig,\
                p_F_loc_mu, p_F_loc_sig,\
                q_z_F_mu, q_z_F_sig,\
                p_z_F_mu, p_z_F_sig\
                = dsarf.forward(mini_batch, u_vals, mini_batch_idxs)
    
            # set gradients to zero in each iteration
                optim_dsarf.zero_grad()
            
            # computing loss
                # excluding missing locations
                idxs_nonnan = ~torch.isnan(mini_batch)
                if args.long:
                    annealing_factor = 0.1 # min(1.0, 0.01 + i / T_A) # inverse temperature
                else:
                    annealing_factor = 0.001
                if args.long:
                    loss_dsarf = ELBO_Loss(mini_batch[idxs_nonnan],
                                          y_hat[idxs_nonnan], 
                                          q_c, p_c,
                                          q_s_0, p_s_0,
                                          q_s_t[:,max(args.lag):-tpred], p_s_t[:,max(args.lag):-tpred],
                                          q_z_0_mus, q_z_0_sigs,
                                          z_0_mu, z_0_sig,
                                          q_z_mus[:,max(args.lag):-tpred], q_z_sigs[:,max(args.lag):-tpred],
                                          p_z_mu[:,:,max(args.lag):-tpred], p_z_sig[:,:,max(args.lag):-tpred],
                                          q_F_loc_mu, q_F_loc_sig,
                                          p_F_loc_mu, p_F_loc_sig,
                                          q_z_F_mu, q_z_F_sig,
                                          p_z_F_mu, p_z_F_sig,
                                          annealing_factor) 
                else:
                    loss_dsarf = ELBO_Loss(mini_batch[idxs_nonnan],
                                          y_hat[idxs_nonnan], 
                                          q_c, p_c,
                                          q_s_0, p_s_0,
                                          q_s_t, p_s_t,
                                          q_z_0_mus, q_z_0_sigs,
                                          z_0_mu, z_0_sig,
                                          q_z_mus, q_z_sigs,
                                          p_z_mu, p_z_sig,
                                          q_F_loc_mu, q_F_loc_sig,
                                          p_F_loc_mu, p_F_loc_sig,
                                          q_z_F_mu, q_z_F_sig,
                                          p_z_F_mu, p_z_F_sig,
                                          annealing_factor)
                
            # back propagation
                loss_dsarf.backward()#(retain_graph = True)
            # update parameters
                optim_dsarf.step()
            # accumulate loss    
                loss_value += loss_dsarf.item()
            
            acc = torch.sum(dsarf.q_c[:,0].argmax(dim=1)==classes).float()/n_data
            time_end = time.time()
            print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
            print('====> Epoch: %d ELBO_Loss : %0.4f Acc: %0.2f'
                  % ((i + 1), loss_value / len(train_loader.dataset), acc))
    
            torch.save(dsarf.state_dict(), PATH_DSARF)
            
            #draw plots once per 10 epochs
            if args.strain and i % 50 == 0:
                plot_result(dsarf, classes, fig_PATH_train,
                            prefix = 'epoch{%.3d}_'%i,
                            ext = ".png", data_st = [dataa, data_mean, data_std],
                            days = t, predict = args.predict,
                            ID = args.ID, spat = spat)
            
    if Restore:
        dsarf.load_state_dict(torch.load(PATH_DSARF,
                                        map_location=lambda storage, loc: storage))
        params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 0}
        train_loader = data.DataLoader(training_set, **params)
        
        for batch_indx, batch_data in enumerate(tqdm(train_loader)):
        
            mini_batch, u_vals, mini_batch_idxs = batch_data
            mini_batch_idxs = mini_batch_idxs.reshape(-1)
            if u_dim == 0:
                u_vals = None
        
            y_hat,\
            q_c, p_c,\
            q_s_0, p_s_0,\
            q_s_t, p_s_t,\
            q_z_0_mus, q_z_0_sigs,\
            z_0_mu, z_0_sig,\
            q_z_mus, q_z_sigs,\
            p_z_mu, p_z_sig,\
            q_F_loc_mu, q_F_loc_sig,\
            p_F_loc_mu, p_F_loc_sig,\
            q_z_F_mu, q_z_F_sig,\
            p_z_F_mu, p_z_F_sig\
            = dsarf.forward(mini_batch, u_vals, mini_batch_idxs)

        plot_result(dsarf, classes, fig_PATH,
                    data_st = [dataa, data_mean, data_std],
                    days = t, ID = args.ID,
                    u_vals = u_vals, spat = spat)

"""
DSARF SETUP & Training--END
#######################################################################
#######################################################################
#######################################################################
#######################################################################
"""