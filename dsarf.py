
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader         
import os
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from tqdm.notebook import tqdm


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
    def __init__(self, z_dim, transition_dim, S, L):
        super(GatedTransition, self).__init__()
        # initialize the linear transformations used in the neural network
        # g (scalar)
        self.fc1_g, self.fc1_g_bias = FC([S, L, z_dim, transition_dim])
        self.fc2_g, self.fc2_g_bias = FC([S, transition_dim, z_dim])
        # nonlinear z transition
        self.fc1_z, self.fc1_z_bias = FC([S, L, z_dim, transition_dim])
        self.fc2_z, self.fc2_z_bias = FC([S, transition_dim, z_dim])
        self.fc3_z, self.fc3_z_bias = FC([S, z_dim, z_dim])
        # linear z transition
        init = [torch.eye(z_dim, z_dim).repeat(S, L, 1, 1),
                torch.zeros(1, z_dim).repeat(S, L, 1, 1)]
        self.fc_z, self.fc_z_bias = FC(init = init)
        # initialize the non-linearities used in the neural network
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t_1):
        """
        Given the latent z_{t-1} corresponding to the time
        step t-1, we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1})
        z is L * Batch * z_dim
        """
        _gate = self.relu(torch.matmul(z_t_1, self.fc1_g) + self.fc1_g_bias) 
        gate = self.sigmoid(torch.matmul(_gate.mean(dim = 1), self.fc2_g) + self.fc2_g_bias)
        # compute the 'proposed mean'
        _z_mean = self.relu(torch.matmul(z_t_1, self.fc1_z) + self.fc1_z_bias) 
        z_mean = torch.matmul(_z_mean.mean(dim = 1), self.fc2_z) + self.fc2_z_bias
        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        z_mean_lin = torch.matmul(z_t_1, self.fc_z) + self.fc_z_bias
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
    def __init__(self, S, factor_dim):
        super(StateTransition, self).__init__()
        # linear s transition
        self.fc_s = nn.Linear(S, S)
        if factor_dim:
            self.fc1_z = nn.Linear(factor_dim, factor_dim)
            self.fc2_z = nn.Linear(factor_dim, factor_dim)
            self.fc3_z = nn.Linear(factor_dim, S)
            self.relu = nn.PReLU()
        # initialize the activation used in the transition
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, s_t_1, z_t_1):
        """
        Given the latent s_{t-1}, we return the probabilities
        that parameterize the cateorical distribution p(s_t | s_{t-1})
        """
        if z_t_1 is None:
            s_t = self.softmax(self.fc_s(s_t_1))
        else:
            s_t = self.relu(self.fc1_z(z_t_1))
            s_t = self.relu(self.fc2_z(s_t))
            s_t = self.softmax(self.fc3_z(s_t))
        return s_t
    

class Emission(nn.Module):
    def __init__(self, factor_dim, D, factorization):
        super(Emission, self).__init__()
        self.factorization = factorization
        if not factorization:
            self.fc_1 = nn.Linear(factor_dim, 2*factor_dim)
            self.fc_2 = nn.Linear(2*factor_dim, 2*factor_dim)
            self.fc_3 = nn.Linear(2*factor_dim, 2*factor_dim)
            self.fc_4 = nn.Linear(2*factor_dim, D)
            self.relu = nn.PReLU()
        else:
            self.fc = nn.Linear(factor_dim, D)

    def forward(self, z_t):
        if not self.factorization:
            y_t = self.relu(self.fc_1(z_t))
            y_t = self.relu(self.fc_2(y_t))
            y_t = self.relu(self.fc_3(y_t))
            y_t = self.fc_4(y_t)
        else:
            y_t = self.fc(z_t)
        return y_t


class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """
    def __init__(self, z_dim, rnn_dim, L):
        super(Combiner, self).__init__()
        # initialize the linear transformations used in the neural network
        self.fc1_z, self.fc1_z_bias = FC([L, z_dim, rnn_dim])
        self.fc2_z = nn.Linear(rnn_dim, z_dim)
        self.fc21_z = nn.Linear(z_dim, z_dim)
        self.fc3_z = nn.Linear(rnn_dim, z_dim)
        self.fc31_z = nn.Linear(z_dim, z_dim)
        # initialize the non-linearities used in the neural network
        self.tanh = nn.PReLU()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, y_{t:T})
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h = torch.matmul(z_t_1, self.fc1_z) + self.fc1_z_bias
        h_combined = 0.5 * (self.tanh(h).mean(dim = 0) + self.tanh(h_rnn))
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.tanh(self.fc2_z(h_combined))
        loc = self.fc21_z(loc)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.tanh(self.fc3_z(h_combined))
        scale = self.fc31_z(scale)
        # return loc, scale which can be fed into Normal
        return loc, scale

class LSTM_obs(nn.Module):
    
    def __init__(self, D, rnn_dim, factor_dim, S):
        super(LSTM_obs, self).__init__()
        self.S = S
        self.rnn = nn.LSTM(D, rnn_dim, 2, batch_first=False,
                           bidirectional=False)
        self.fc1_rnn = nn.Linear(rnn_dim, rnn_dim)
        self.fc2_rnn = nn.Linear(rnn_dim, rnn_dim)
        self.fc3_rnn = nn.Linear(rnn_dim, factor_dim*2)
        if S:
            self.fc1_rnn_s = nn.Linear(rnn_dim, rnn_dim)
            self.fc2_rnn_s = nn.Linear(rnn_dim, rnn_dim)
            self.fc3_rnn_s = nn.Linear(rnn_dim, S)
            
        self.relu = nn.PReLU()
    def forward(self, x):
        rnn_output, _= self.rnn(x)
        z = self.relu(self.fc1_rnn(rnn_output))
        z = self.relu(self.fc2_rnn(z))
        z = self.fc3_rnn(z)
        s = None
        if self.S:
            s = self.relu(self.fc1_rnn_s(rnn_output))
            s = self.relu(self.fc2_rnn_s(s))
            s = self.fc3_rnn_s(s).permute(1,0,2)
            
        return z, s
  
    
    
    def __init__(self):
        super(CNN, self).__init__()
        #self.rnn = nn.LSTM(factor_dim, d_h, 4, batch_first=True, bidirectional=False) 
        self.conv1 = nn.Conv1d(factor_dim, 25, 5, 1)
        self.bn1 = nn.BatchNorm1d(25)
        self.conv2 = nn.Conv1d(25, 15, 5, 2)
        self.bn2 = nn.BatchNorm1d(15)
        self.conv3 = nn.Conv1d(15, 5, 5, 2)
        self.bn3 = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(45, 10)
        self.fc2 = nn.Linear(10, labels.max().item()+1)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: N x T x D
        x = self.relu(self.conv1(x.permute(0,2,1)))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.relu(self.fc1(x.reshape(x.shape[0],-1)))
        x = self.fc2(x)
        return xz

class DSARF(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution for the Deep Markov Factor Analysis
    """
    def __init__(self, D, factor_dim, L, S, transition_dim=None,
                 VI = {'rnn_dim': None, 'combine': False, 'S': False}, recurrent = False,
                 recursive_state = False,
                 factorization = True, lr = 1e-2, batch_size = 20):
        super().__init__()
        
        self.D, self.factor_dim, self.L, self.S = D, factor_dim, L, S
        transition_dim = [transition_dim if transition_dim is not None else factor_dim][0]
        self.VI, self.recurrent, self.recursive_state = VI, recurrent, recursive_state
        # instantiate pytorch modules used in the model and guide below
        self.trans = GatedTransition(factor_dim, transition_dim, S, len(L))
        self.strans = StateTransition(S, [factor_dim if recurrent else 0][0])
        
        if VI['rnn_dim'] is not None:
            self.lstm_obs = LSTM_obs(D, VI['rnn_dim'], factor_dim, [S if VI['S'] else 0][0])
            if VI['combine']:
                self.combiner = Combiner(factor_dim, factor_dim*2, len(L))

        self.p_s_0 = nn.Parameter(torch.ones(1, S))
        self.z_0_mu = nn.Parameter(torch.rand(max(L), 1, factor_dim)- 1/2)
        self.z_0_sig = nn.Parameter((torch.ones(max(L), 1, factor_dim) / 2 * 0.15 * 5).log())
        
        #self.q_F_loc_mu = nn.Parameter(torch.rand(1, factor_dim, D)- 1/2)
        self.emission = Emission(factor_dim, D, factorization)
        self.mean, self.std, self.grad = 0, 1, True
        self.lr, self.batch_size = lr, batch_size
    
    def fit(self, data, epoch_num = 500):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        if self.grad:
            data_st_cat = np.concatenate(data, axis = 0)
            self.mean = data_st_cat[~np.isnan(data_st_cat)].mean()
            self.std = data_st_cat[~np.isnan(data_st_cat)].std()
        
        dataa_train = [(data[i] - self.mean)/self.std for i in range(len(data))]
        
        # set parameters 
        n_data = len(dataa_train)
        lens = [len(i) for i in dataa_train]
        #form data for training
        training_set_part = [(torch.FloatTensor(y),torch.LongTensor([i])) for i, y in enumerate(dataa_train)]
    
        # initialize model
        for p in self.parameters(): #turn gradients on/off
            p.requires_grad  = self.grad
        dsarf = self.DSARF_(self, n_data, lens).to(device)
        
        optim_dsarf = optim.Adam(dsarf.parameters(), lr = self.lr)
        # number of parameters  
        total_params = sum(p.numel() for p in dsarf.parameters())
        learnable_params = sum(p.numel() for p in dsarf.parameters() if p.requires_grad)
        print('Total Number of Parameters: %d' % total_params)
        print('Learnable Parameters: %d' %learnable_params)
        
        params = {'batch_size': self.batch_size,
                  'shuffle': True,
                  'num_workers': 0}
        train_loader = DataLoader(training_set_part, **params)
        
        for i in tqdm(range(epoch_num)):
            #time_start = time.time()
            loss_value = 0.0
            for batch_indx, batch_data in enumerate(train_loader):
            # update DSARF
                mini_batch, mini_batch_idxs = batch_data
                mini_batch_idxs = mini_batch_idxs.reshape(-1)
                mini_batch = mini_batch.to(device)
                mini_batch_idxs = mini_batch_idxs.to(device)
    
                y_hat,\
                q_s_0, p_s_0,\
                q_s_t, p_s_t,\
                q_z_0_mus, q_z_0_sigs,\
                z_0_mu, z_0_sig,\
                q_z_mus, q_z_sigs,\
                p_z_mu, p_z_sig\
                = dsarf.forward(mini_batch, mini_batch_idxs)
    
                # set gradients to zero in each iteration
                optim_dsarf.zero_grad()
            
                # computing loss
                idxs_nonnan = ~torch.isnan(mini_batch)
                annealing_factor = 0.001
                
                loss_dsarf = ELBO_Loss(mini_batch[idxs_nonnan],
                                      y_hat[idxs_nonnan],
                                      q_s_0, p_s_0,
                                      q_s_t[:, max(self.L):], p_s_t[:, max(self.L):],
                                      q_z_0_mus, q_z_0_sigs,
                                      z_0_mu, z_0_sig,
                                      q_z_mus[:, max(self.L):], q_z_sigs[:, max(self.L):],
                                      p_z_mu[:,:, max(self.L):], p_z_sig[:,:, max(self.L):],
                                      annealing_factor)
                
                # back propagation
                loss_dsarf.backward()
                # update parameters
                optim_dsarf.step()
                # accumulate loss  
                loss_value += loss_dsarf.item()
            
            #time_end = time.time()
            #print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
            if (i % 50 == 0) or (i == epoch_num - 1):
                NRMSE = dsarf.report_stats(data)
                epoch = i + 1
                
            print('ELBO_Loss: %0.4f, Epoch %d: {NRMSE_recv : %0.2f, NRMSE_pred : %0.2f}'
                  % (loss_value / len(train_loader.dataset),
                     epoch, NRMSE['NRMSE_recv'], NRMSE['NRMSE_pred']),
                  end="\r", flush=True)
            #torch.save(dsarf.state_dict(), PATH_DSARF)
        return dsarf
        
    def infer(self, data, epoch_num = 500):
        self.grad = False
        if self.VI['rnn_dim'] is not None:
            epoch_num = 1
        dsarf = self.fit(data, epoch_num)
        self.grad = True
        return dsarf
         
    class DSARF_(nn.Module):
        def __init__(self, dsarf, n_data, lens):
            super().__init__()
            
            self.dsarf = dsarf
            self.lens = lens
            T = max(lens) # use maximum T to conveniently support varying length
            self.softmax = nn.Softmax(dim = -1)
            self.q_s = nn.Parameter(torch.ones(n_data, T, dsarf.S) / dsarf.S, requires_grad=False)
            self.q_s_0 = nn.Parameter(torch.ones(n_data, dsarf.S))
            
            self.q_z_0_mu = nn.Parameter(torch.rand(n_data, max(dsarf.L), dsarf.factor_dim)- 1/2)
            self.q_z_0_sig = nn.Parameter((torch.ones(n_data, max(dsarf.L), dsarf.factor_dim) / 2 * 0.1).log())
            
            self.q_z_mu = nn.Parameter(torch.rand(n_data, T, dsarf.factor_dim)- 1/2)
            self.q_z_sig = nn.Parameter((torch.ones(n_data, T, dsarf.factor_dim) / 2 * 0.1).log())
            if dsarf.VI['rnn_dim'] is not None: 
                self.q_z_mu.requires_grad, self.q_z_sig.requires_grad = False, False
    
        def Reparam(self, mu_latent, sigma_latent):
            eps = mu_latent.data.new(mu_latent.size()).normal_()
            return eps.mul(sigma_latent.exp()).add_(mu_latent)
        
        # the model p(y|w,F)p(w|z)p(z_t|z_{t-1},u_{t-1})p(z_0|c)p(c)p(F|z_F)p(z_F)
        def forward(self, mini_batch, mini_batch_idxs):
            # z_values = (data_points, time_points + max(L), z_dim)
            # F_loc_values = (factor_dim, D)
            N = mini_batch.size(0)
            T_b = mini_batch.size(1)
            z_dim = self.q_z_0_mu.size(-1)
            
            q_z_0_mus = self.q_z_0_mu[mini_batch_idxs] #batch*L*z_dim
            q_z_0_sigs = self.q_z_0_sig[mini_batch_idxs] #batch*L*z_dim
            z_0_values = self.Reparam(q_z_0_mus, q_z_0_sigs)
            
            if self.dsarf.VI['rnn_dim'] is not None:
                
                y_filled = self.dsarf.emission(self.q_z_mu[mini_batch_idxs, :T_b])
                idxs_nans = torch.isnan(mini_batch)
                obs = torch.zeros_like(mini_batch)
                obs[~idxs_nans] = mini_batch[~idxs_nans] * 1.0
                obs[idxs_nans] = y_filled[idxs_nans].data * 1.0

                rnn_output, q_s_t = self.dsarf.lstm_obs(obs.permute(1,0,2))
                
                if self.dsarf.VI['combine']:
                    q_z_mus = torch.Tensor([]).reshape(N, 0, z_dim).to(rnn_output.device)
                    q_z_sigs = torch.Tensor([]).reshape(N, 0, z_dim).to(rnn_output.device)
                    z_values = z_0_values.clone()
                    z_prev = z_values.permute(1,0,2)[-np.array(self.dsarf.L)]
                    for i in range(T_b):
                        loc, scale = self.dsarf.combiner(z_prev, rnn_output[i])
                        z_val = self.Reparam(loc, scale)
                        z_values = torch.cat((z_values,z_val.unsqueeze(1)), dim=1)
                        z_prev = z_values.permute(1,0,2)[-np.array(self.dsarf.L)]
                        q_z_mus = torch.cat((q_z_mus,loc.unsqueeze(1)), dim=1)
                        q_z_sigs = torch.cat((q_z_sigs,scale.unsqueeze(1)), dim=1) 
                else:
                    q_z_mus = rnn_output.permute(1,0,2)[:,:,:z_dim] #batch*T*z_dim
                    q_z_sigs = rnn_output.permute(1,0,2)[:,:,z_dim:] #batch*T*z_dim
                    z_t_values = self.Reparam(q_z_mus, q_z_sigs)
                    z_values = torch.cat((z_0_values, z_t_values), dim = 1) 
                self.q_z_mu[mini_batch_idxs, :T_b] = q_z_mus.detach()
                self.q_z_sig[mini_batch_idxs, :T_b] = q_z_sigs.detach()
            else:
                q_z_mus = self.q_z_mu[mini_batch_idxs, :T_b] #batch*T*z_dim
                q_z_sigs = self.q_z_sig[mini_batch_idxs, :T_b] #batch*T*z_dim
                z_t_values = self.Reparam(q_z_mus, q_z_sigs)
                z_values = torch.cat((z_0_values, z_t_values), dim = 1)   
                  
            # p(z_t|z_{t-1},u{t-1}, s_t) = Normal(z_loc, z_scale)
            z_t_1 = torch.Tensor([]).reshape(0, N * T_b, z_dim).to(z_values.device)
            for lag in self.dsarf.L:
                z_t_1 = torch.cat((z_t_1,
                                   z_values[:, max(self.dsarf.L)-lag:-lag].reshape(1, -1, z_dim)),
                                  dim = 0)

            p_z_mu, p_z_sig = self.dsarf.trans(z_t_1)
            p_z_mu = p_z_mu.view(self.dsarf.S, N, T_b, -1)
            p_z_sig = p_z_sig.view(self.dsarf.S, N, T_b, -1)
            
            # compute q(s_0)
            p_s_0 = self.softmax(self.dsarf.p_s_0)
            q_s_0 = self.softmax(self.q_s_0[mini_batch_idxs])
            # compute q(s_t) = p(s_t|z_t) = p(z_t|z_{t-1},s_t)p(s_t|s_{t-1})
            if not self.dsarf.VI['S']:
                q_s_t = self.q_s[mini_batch_idxs, :T_b]
            else:
                #krnl = 5
                #q_s_t = torch.cat((q_s_t, q_s_t[:,-1:].repeat(1, krnl-1, 1)), 1).unfold(1, krnl, 1).mean(-1)
                q_s_t = self.softmax(q_s_t)
            q_s = torch.cat((q_s_0.unsqueeze(1), q_s_t), dim=1)
            p_s_t = self.dsarf.strans(q_s[:,:-1],
                                [z_values[:,max(self.dsarf.L)-1:-1]
                                 if self.dsarf.recurrent else None][0])
            z_t_vals = z_values[:, max(self.dsarf.L):] # batch*T_b*z_dim
            
            # p(y|z,F) = Normal(z*F, sigma)

            y_hat = self.dsarf.emission(z_values[:, max(self.dsarf.L):]) # S*N*T*D
            
            # compute q(s_t)
            if not self.dsarf.VI['S']:
                if not self.dsarf.recursive_state:
                    q_s_t = (p_s_t.permute(2, 0, 1)+1e-4).log()\
                              -1/2*((z_t_vals - p_z_mu)\
                              /(p_z_sig.exp()+1e-4)).pow(2).sum(dim = -1)\
                              -p_z_sig.sum(dim = -1)  # n*T*K, S*n*T*K = S*n*T
                    #krnl = 5
                    q_s_t = q_s_t.permute(1, 2, 0)
                    #q_s_t = torch.cat((q_s_t, q_s_t[:,-1:].repeat(1, krnl-1, 1)), 1).unfold(1, krnl, 1).mean(-1)
                    q_s_t = self.softmax(q_s_t)
            
                else:
                    # compute q(s_t) = p(s_t|z_t) = p(z_t|z_{t-1},s_t)p(s_t|s_{t-1})
                    p_s_t = torch.Tensor([]).reshape(N, 0, self.dsarf.S)
                    q_s_t = torch.Tensor([]).reshape(N, 0, self.dsarf.S)
                    s_t_1 = q_s_0.clone()
                    for i in range(T_b):
                        # p(s_t|s_{t-1})
                        p_s = self.dsarf.strans(s_t_1,
                                          [z_values[:,i+max(self.dsarf.L)-1]
                                           if self.dsarf.recurrent else None][0]) # batch*S
                        p_s_t = torch.cat((p_s_t, p_s.unsqueeze(1)), dim = 1)
                        z_t_vals = z_values[:, i + max(self.dsarf.L)] # batch*z_dim
                        # compute q(s_t)
                        q_s = (p_s.permute(1, 0)+1e-4).log()\
                              -1/2*((z_t_vals - p_z_mu[:,:,i])\
                              /(p_z_sig[:,:,i].exp()+1e-4)).pow(2).sum(dim = -1)\
                              -p_z_sig[:,:,i].sum(dim = -1)
                        s_t_1 = self.softmax(q_s.permute(1, 0))
                        q_s_t = torch.cat((q_s_t, s_t_1.unsqueeze(1)), dim = 1)
    
            self.q_s[mini_batch_idxs, :T_b] = q_s_t.detach()
            if self.dsarf.S == 1:
                q_s_t = torch.ones(N, T_b, 1).to(y_hat.device)
            
            q_z_0_mus.unsqueeze_(2)
            q_z_0_sigs.unsqueeze_(2)
            
            return y_hat,\
                    q_s_0, p_s_0,\
                    q_s_t, p_s_t,\
                    q_z_0_mus, q_z_0_sigs,\
                    self.dsarf.z_0_mu, self.dsarf.z_0_sig,\
                    q_z_mus, q_z_sigs,\
                    p_z_mu, p_z_sig
                    
        def report_stats(self, data):
            
            y_recv = self.dsarf.emission(self.q_z_mu).detach().cpu().numpy()*self.dsarf.std + self.dsarf.mean
            y_pred , _, _ = self.short_predict()
            
            NRMSE = [compute_NRMSE(data, y_recv), compute_NRMSE(data, y_pred)]
            NRMSE = dict(zip(['NRMSE_recv','NRMSE_pred'], NRMSE))
            return NRMSE
            
        
        def short_predict(self, s=None):
            
            N, T_b, z_dim = self.q_z_mu.shape
            # p(z_t|z_{t-1}, s_t) = Normal(z_loc, z_scale)
            z_t_1 = torch.Tensor([]).reshape(0, N * (T_b-max(self.dsarf.L)), z_dim).to(self.q_z_mu.device)
            for lag in self.dsarf.L:
                z_t_1 = torch.cat((z_t_1,
                                   self.q_z_mu[:, max(self.dsarf.L)-lag:-lag].reshape(1, -1, z_dim)),
                                  dim = 0)
            p_z_mu, p_z_sig = self.dsarf.trans(z_t_1)
            p_z_mu = p_z_mu.view(self.dsarf.S, N, T_b-max(self.dsarf.L), -1)
            p_z_sig = p_z_sig.view(self.dsarf.S, N, T_b-max(self.dsarf.L), -1)
            if s is not None:
                p_z_mu = p_z_mu[[s]]
                p_z_sig = p_z_sig[[s]]
            p_s_t = self.dsarf.strans(self.q_s[:,max(self.dsarf.L)-1:-1],
                                [self.q_z_mu[:,max(self.dsarf.L)-1:-1]
                                 if self.dsarf.recurrent else None][0])
            
            z_val_p = torch.cat(((self.q_z_mu+self.q_z_sig.exp())[:,:max(self.dsarf.L)],
                                 (p_s_t.permute(2,0,1).unsqueeze(-1) * (p_z_mu+p_z_sig.exp())).sum(dim=0)), dim=1)
            z_val_n = torch.cat(((self.q_z_mu-self.q_z_sig.exp())[:,:max(self.dsarf.L)],
                                 (p_s_t.permute(2,0,1).unsqueeze(-1) * (p_z_mu-p_z_sig.exp())).sum(dim=0)), dim=1)
            z_val = torch.cat((self.q_z_mu[:,:max(self.dsarf.L)],
                               (p_s_t.permute(2,0,1).unsqueeze(-1) * p_z_mu).sum(dim=0)), dim=1)
            
            y_pred_n = self.dsarf.emission(z_val_n).detach().cpu().numpy()*self.dsarf.std+self.dsarf.mean 
            y_pred_n = [j[:self.lens[i]] for i, j in enumerate(y_pred_n)]
            y_pred_p = self.dsarf.emission(z_val_p).detach().cpu().numpy()*self.dsarf.std+self.dsarf.mean 
            y_pred_p = [j[:self.lens[i]] for i, j in enumerate(y_pred_p)]
            y_pred = self.dsarf.emission(z_val).detach().cpu().numpy()*self.dsarf.std+self.dsarf.mean 
            y_pred = [j[:self.lens[i]] for i, j in enumerate(y_pred)]
            return y_pred, y_pred_n, y_pred_p
        
        

        def long_predict(self, steps, s = None):
            
            z_values = z_values_p = z_values_n = self.q_z_mu[:,-max(self.dsarf.L):]
            z_t_1 = z_values.permute(1,0,2)[-np.array(self.dsarf.L)]
            z_t_1_s = z_values.permute(1,0,2)[-1]
    
            s_vals = self.q_s[:,-max(self.dsarf.L):]
            s_t_1 = self.q_s[:, -1]
            for i in range(steps):
                p_z_mu, p_z_sig = self.dsarf.trans(z_t_1) # S*N*z_dim
                p_s = self.dsarf.strans(s_t_1,
                                        [z_t_1_s if self.dsarf.recurrent else None][0]) # N * S
                if s is not None:
                    z_val = p_z_mu[s]
                    z_val_p = p_z_mu[s]+p_z_sig[s].exp()
                    z_val_n = p_z_mu[s]-p_z_sig[s].exp()
                else:
                    z_val = (p_s.permute(1,0).unsqueeze(-1) * p_z_mu).sum(0)
                    z_val_p = (p_s.permute(1,0).unsqueeze(-1) * (p_z_mu+p_z_sig.exp())).sum(0)
                    z_val_n = (p_s.permute(1,0).unsqueeze(-1) * (p_z_mu-p_z_sig.exp())).sum(0)
                z_values = torch.cat((z_values, z_val.unsqueeze(1)), dim = 1)
                z_values_p = torch.cat((z_values_p, z_val_p.unsqueeze(1)), dim = 1)
                z_values_n = torch.cat((z_values_n, z_val_n.unsqueeze(1)), dim = 1)
                z_t_1 = z_values.permute(1,0,2)[-np.array(self.dsarf.L)]
                z_t_1_s = z_values.permute(1,0,2)[-1]
                s_vals = torch.cat((s_vals, p_s.unsqueeze(1)), dim = 1)
                s_t_1 = p_s * 1.0
                
            y_pred_n = self.dsarf.emission(z_values_n[:, max(self.dsarf.L):]).detach().cpu().numpy()*self.dsarf.std+self.dsarf.mean 
            y_pred_n = [j[:self.lens[i]] for i, j in enumerate(y_pred_n)]
            y_pred_p = self.dsarf.emission(z_values_p[:, max(self.dsarf.L):]).detach().cpu().numpy()*self.dsarf.std+self.dsarf.mean 
            y_pred_p = [j[:self.lens[i]] for i, j in enumerate(y_pred_p)]
            y_pred = self.dsarf.emission(z_values[:, max(self.dsarf.L):]).detach().cpu().numpy()*self.dsarf.std+self.dsarf.mean 
            y_pred = [j[:self.lens[i]] for i, j in enumerate(y_pred)]
                
             
            return y_pred, y_pred_n, y_pred_p


            
        def plot_predict(self, data, steps = None, path = './plots/'):
            if not os.path.exists(path):
                os.makedirs(path)
            if steps is None:
                y_pred, y_pred_n, y_pred_p = self.short_predict()
            else:
                y_pred, y_pred_n, y_pred_p = self.long_predict(steps)
            for j in range(0 , len(y_pred), max(len(y_pred)//4, 1)):
                idx_locs = [i for i in range(0, self.dsarf.D, max(self.dsarf.D//5, 1))]
                fig = plt.figure(figsize=(10,7/3*len(idx_locs)))
                for i, idx_loc in enumerate(idx_locs):
                    ax = fig.add_subplot(len(idx_locs),1,i+1)
                    ax.plot(data[j][:,idx_loc], label = "Actual")
                    y_preds_p = y_pred_p[j][:, idx_loc]
                    y_preds_n = y_pred_n[j][:,idx_loc]
                    y_preds = y_pred[j][:, idx_loc]
                    ax.plot(y_preds, 'r-.', label = "Predicted", alpha = 0.8)
                    ax.fill_between(np.arange(len(y_preds)), y_preds_n, y_preds_p, color = 'red', alpha=0.1)
                    ax.legend(framealpha = 0, fontsize=13)
                    ax.set_ylabel('loc #%d' %idx_loc, fontsize=13)
                    ax.set_xlabel('Time', fontsize=13)
                plt.tight_layout()
                plt.show()
                fig.savefig(path + "prediction_%d.png" %j, bbox_inches='tight')
                plt.close()

        def plot_states(self, index = None, k_smooth = None, path = './plots/'):
            if not os.path.exists(path):
                os.makedirs(path)        
            import seaborn as sns
            sns.set_style("white")
            sns.set_context("talk")
            color_names = ["windows blue","red","amber","faded green","dusty purple",
                           "orange","clay","pink","greyish","mint","cyan",
                           "steel blue","forest green","pastel purple",
                           "salmon","dark brown","fuchsia","crimson",
                           "chocolate","lime"]
            colors = sns.xkcd_palette(color_names)
            from matplotlib.colors import ListedColormap
            cmap_limited = ListedColormap(colors[:self.dsarf.S])
            s_vals = self.q_s.argmax(-1).detach().cpu().numpy().astype('float')
            if index is None:
                idxs = [i for i in range(0, len(s_vals), max(len(s_vals)//10,1))]
            else:
                idxs = [index]
            for idx in idxs:
                s_vals[idx, self.lens[idx]:] = np.nan
            if k_smooth is not None:
                from scipy.signal import medfilt
                s_vals = medfilt(s_vals, [1, k_smooth])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(s_vals[idxs], aspect='auto', cmap=cmap_limited)
            ax.set_yticks([])
            ax.tick_params(axis='both', which='major', labelsize=21)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time', fontsize=21)
            ax.set_ylabel('Sample', fontsize=21)
            plt.show()
            fig.savefig(path+'States.png', bbox_inches='tight')


                   
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


def ELBO_Loss(mini_batch, y_hat,\
              q_s_0, p_s_0,\
              q_s_t, p_s_t,\
              q_z_0_mus, q_z_0_sigs,\
              z_0_mu, z_0_sig,\
              q_z_mus, q_z_sigs,\
              p_z_mu, p_z_sig,\
              annealing_factor = 1):
    
    # y_hat: S*N*T*D, mini_batch = N*T*D
    rec_loss = (y_hat - mini_batch).pow(2).sum()

    KL_s_0 = KLD_Cat(q_s_0.mean(dim=0), p_s_0).sum()
    KL_s_t = KLD_Cat(q_s_t, p_s_t).sum()
    KL_z_0 = KLD_Gaussian(q_z_0_mus, q_z_0_sigs,
                             z_0_mu, z_0_sig).sum()
    KL_z = (q_s_t.permute(2,0,1) *
                  KLD_Gaussian(q_z_mus, q_z_sigs, 
                               p_z_mu, p_z_sig)).sum()

    
    return rec_loss +annealing_factor * (KL_s_0 + KL_s_t
                                          + KL_z_0 + KL_z)


def compute_NRMSE(y, y_hat):
    idxs = [(len(i), ~np.isnan(i)) for i in y]
    RMSE = [np.power(y[i][idxs[i][1]] - y_hat[i][:idxs[i][0]][idxs[i][1]],2) for i in range(len(y))]
    RMSE = np.sqrt(sum([i.sum() for i in RMSE])/sum([len(i) for i in RMSE]))
    power = [y[i][idxs[i][1]]**2 for i in range(len(y))]
    NRMSE = RMSE/np.sqrt(sum([i.sum() for i in power])/sum([len(i) for i in power]))*100
    return NRMSE


# root_dir = "C:/Users/Amir/Downloads/DSARF_bat/data/bat.json"

# import json
# f = open(root_dir)
# data_st = json.load(f)
# D = len(data_st['joints']) * 3
# data_st = [np.array(data_st[key]).reshape(-1, D)
#            for key in list(data_st.keys())[3:-2]]
                
# dsarf = DSARF(D, factor_dim =5, L=[1], S=2, batch_size=1, recurrent=True)

# model = dsarf.fit(data_st, 200)
# model.plot_states()
# model.plot_predict(data_st)
# model.report_stats(data_st)

