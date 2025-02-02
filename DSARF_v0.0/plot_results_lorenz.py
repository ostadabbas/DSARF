

import numpy as np
import torch
import matplotlib
#matplotlib.use('Agg')
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import pdb

def plot_result(dmfa, classes,
                fig_PATH, prefix = '',
                ext = ".png", data_st = None,
                days = None, predict = True, ID = None,
                u_vals=None, spat=None):
    if ID is not None:
        ID_name = np.array(['bat','guangzhou','birmingham','hangzhou','seattle'])
        data_id = np.where(ID_name == ID)[0][0]
    
    n_class = dmfa.p_c.size(-1)
    S = dmfa.S
    T = dmfa.q_z_mu.size(1)
    D = dmfa.q_F_loc_mu.shape[-1]
    z_0 = dmfa.q_z_0_mu.detach().numpy()
    z_0_p = dmfa.z_0_mu.detach().numpy()
    z_0_p_sig = dmfa.z_0_sig.exp().detach().numpy()
    fig = plt.figure()
    colors = ['b','r','g','y']
    labels = ['group%d'%(c+1) for c in range(n_class)]
    c_idx = classes.detach().numpy()
    for i in range(len(dmfa.L)):
        ax = fig.add_subplot(1, len(dmfa.L), i+1)
        ax.set_title("$z_{-%d}$" %dmfa.L[i])
        for j in range(n_class):
            ax.scatter(z_0[c_idx==j,-dmfa.L[i],0],z_0[c_idx==j,-dmfa.L[i],1], label = labels[j])
            circle = Ellipse((z_0_p[-dmfa.L[i],j, 0], z_0_p[-dmfa.L[i],j, 1]),
                             z_0_p_sig[-dmfa.L[i],j,0]*2, z_0_p_sig[-dmfa.L[i],j,1]*2,
                             color=colors[j], alpha = 0.2)
            ax.add_artist(circle)
        ax.legend()
#        plt.tick_params(
#                axis='both',          # changes apply to the x-axis
#                which='both',      # both major and minor ticks are affected
#                bottom=False,      # ticks along the bottom edge are off
#                top=False,         # ticks along the top edge are off
#                labelbottom=False,
#                right=False, left=False, labelleft=False) # labels along the bottom edge are off
    fig.savefig(fig_PATH + "%sq_z_lag" %prefix + ext)
    zs_p = (dmfa.q_z_mu+1*dmfa.q_z_sig.exp()).detach().numpy()
    zs_n = (dmfa.q_z_mu-1*dmfa.q_z_sig.exp()).detach().numpy()
    zs = dmfa.q_z_mu.detach().numpy()
    s_idx = dmfa.q_s.argmax(dim=-1).detach().numpy()
    s_vals = dmfa.q_s.max(dim=-1)[0].detach().numpy()
    labels = ['state%d'%(c+1) for c in range(S)]
    colors  = plt.cm.jet(np.linspace(0,1,S))
    for j in range(0, T, T//10):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("z_%d" %j)
        for k in range(S):
            idx = s_idx[:,j]==k
            if idx.sum() != 0:
                fig_color = np.tile(colors[k], (idx.sum(),1))
                fig_color[:,3] = s_vals[idx, j]
                ax.scatter(zs[idx,j,0],zs[idx,j,1],
                            label = labels[k], color = fig_color)
#        plt.tick_params(
#        axis='both',          # changes apply to the x-axis
#        which='both',      # both major and minor ticks are affected
#        bottom=False,      # ticks along the bottom edge are off
#        top=False,         # ticks along the top edge are off
#        labelbottom=False,
#        right=False, left=False, labelleft=False) # labels along the bottom edge are off
        ax.legend()
        fig.savefig(fig_PATH + "%sq_z_%d" %(prefix,j) + ext)
    
    s_0 = dmfa.q_s_0.argmax(dim=-1).detach().numpy()
    ss = np.concatenate((np.expand_dims(s_0, 1), s_idx), axis = 1)
    fig = plt.figure(figsize=(20,3))
    idx = [k for k in range(0, ss.shape[0], ss.shape[0]//4 + 1)]
    cnt = 1
    for k in idx: #plot at most for 4 data points
        ax = fig.add_subplot(len(idx), 1, cnt)
        cnt += 1
        ax.set_title("State transitions data #%d" %k)
        ax.step(np.arange(0, T+1)-1/2, ss[k])
#        plt.tick_params(
#            axis='both',          # changes apply to the x-axis
#            which='both',      # both major and minor ticks are affected
#            bottom=False,      # ticks along the bottom edge are off
#            top=False,         # ticks along the top edge are off
#            labelbottom=False,
#            right=False, left=False, labelleft=False) # labels along the bottom edge are off
    plt.tight_layout()
    fig.savefig(fig_PATH + "%sstate_trajectory" %prefix + ext)
    
    dataa, data_mean, data_std = data_st
    dataa = [j*data_std[i]+data_mean[i] for i, j in enumerate(dataa)]
    
    N = len(dataa)
    T_b = len(dataa[-1])
    ws = zs[-1,:T_b]
    ws = np.concatenate((ws, zs[-1,:max(dmfa.L)]), axis = 0) #added recently
    z_values = dmfa.q_z_mu[-1:,:max(dmfa.L)] #dmfa.q_z_0_mu[-1:] #edited recently
    z_t_1 = z_values.permute(1,0,2)[-np.array(dmfa.L)]
    z_t_1_s = z_values.permute(1,0,2)[-1]
    s_t_1 = dmfa.q_s[-1:, max(dmfa.L) - 1] #torch.nn.Softmax(dim=-1)(dmfa.q_s_0[-1:]) #edited
    for i in range(T_b-max(dmfa.L)):#range(T_b): #edited recently
        if u_vals is not None and max(dmfa.L) == 1:
            u_t_1 = u_vals[-1:,i].unsqueeze(0)
        else:
            u_t_1 = None
        p_s = dmfa.strans(s_t_1, z_t_1_s) # 1 * S
        p_z_mu, p_z_sig = dmfa.trans(z_t_1, u_t_1) # S*1*z_dim
        p_z_mu = dmfa.Reparam(p_z_mu, p_z_sig)
        z_val = (p_s.reshape(-1, 1, 1) * p_z_mu).sum(dim=0)
        z_values = torch.cat((z_values, z_val.unsqueeze(0)), dim = 1)
        z_t_1 = z_values.permute(1,0,2)[-np.array(dmfa.L)]
        z_t_1_s = z_values.permute(1,0,2)[-1]
        ws = np.concatenate((ws, z_val.detach().numpy()), axis = 0)
        s_t_1 = p_s * 1.0
    f_locs = dmfa.q_F_loc_mu.detach().numpy()
    y_pred = np.matmul(ws, f_locs)*data_std[-1]+data_mean[-1]
    if ID is not None:
        if data_id == 0:
            I = 3
    else:
        I = 1
    D = D//I
    y_pred = y_pred.reshape(-1, D, I)
    idxs = [(len(i), ~np.isnan(i)) for i in dataa]
    
    plt.close('all')
    if predict or days == 0:    
        if ID is not None:
            titles = ['Joint',
                      'Guangzhou road segment',
                      'Birmingham car park',
                      'Hangzhou metro staion',
                      'Seattle loop detector']
            ylabels = [['Location x', 'Location y', 'Location z'],
                       ['Traffic speed'],
                       ['Occupancy'],
                       ['Passenger flow'],
                       ['Traffic speed']]    
        fontsize= 12
        for idx_loc in range(0, D, D//5):
            fig = plt.figure(figsize=(10,7))
            for i in range(I):
                ax = fig.add_subplot(I,1,i+1)
                ax.plot(dataa[-1].reshape(-1, D, I)[:,idx_loc, i], label = "Actual")
                ax.plot(y_pred[:T_b,idx_loc, i], 'r-',label = "Recovered", alpha = 0.8)
                y_preds = y_pred[:,idx_loc, i] * 1.0
                y_preds[:T_b] = np.nan
                ax.plot(y_preds, 'r-.', label = "Predicted", alpha = 0.8)
                ax.legend(framealpha = 0, fontsize=13)
                if ID is not None:
                    ax.set_title(titles[data_id]+' #%d'%idx_loc, fontsize=fontsize)
                    ax.set_ylabel(ylabels[data_id][i], fontsize=fontsize+2)
            plt.tight_layout()
            fig.savefig(fig_PATH + "%sprediction_long_term%s" %(prefix,idx_loc) + ext, bbox_inches='tight')
        plt.close('all')

    y_pred = np.matmul(zs, f_locs)
    y_pred = np.asarray([j*data_std[i]+data_mean[i] for i, j in enumerate(y_pred)])
    if  predict or days == 0:
        if days == 0:
            days = N
        RMSE = [np.power(dataa[i][idxs[i][1]] - y_pred[i,:idxs[i][0]][idxs[i][1]],2) for i in range(-days,0)]
        RMSE = np.sqrt(sum([i.sum() for i in RMSE])/sum([len(i) for i in RMSE]))
        print('Test RMSE %.2f' %RMSE)
        MAPE = [np.absolute((dataa[i][idxs[i][1]] - y_pred[i,:idxs[i][0]][idxs[i][1]])/dataa[i][idxs[i][1]]) for i in range(-days,0)]
        MAPE = sum([i.sum() for i in MAPE])/sum([len(i) for i in MAPE])*100
        print('Test MAPE %.2f' %MAPE)
        NRMSE = [dataa[i][idxs[i][1]]**2 for i in range(-days,0)]
        NRMSE = RMSE/np.sqrt(sum([i.sum() for i in NRMSE])/sum([len(i) for i in NRMSE]))*100
        print('Test NRMSE %.2f' %NRMSE)
    if predict == False and days != N:
        RMSE = [np.power(dataa[i][idxs[i][1]] - y_pred[i,:idxs[i][0]][idxs[i][1]],2) for i in range(N-days)]
        RMSE = np.sqrt(sum([i.sum() for i in RMSE])/sum([len(i) for i in RMSE]))
        print('Train RMSE %.2f' %RMSE)
        MAPE = [np.absolute((dataa[i][idxs[i][1]] - y_pred[i,:idxs[i][0]][idxs[i][1]])/dataa[i][idxs[i][1]]) for i in range(N-days)]
        MAPE = sum([i.sum() for i in MAPE])/sum([len(i) for i in MAPE])*100
        print('Train MAPE %.2f' %MAPE)
        NRMSE = [dataa[i][idxs[i][1]]**2 for i in range(N-days)]
        NRMSE = RMSE/np.sqrt(sum([i.sum() for i in NRMSE])/sum([len(i) for i in NRMSE]))*100
        print('Train NRMSE %.2f' %NRMSE)
        
    if predict or days == N:
#        if days == N and ID is None:
#            days = 2
        y_recv_p = np.matmul(zs_p, f_locs)
        y_recv_p = [j*data_std[i]+data_mean[i] for i, j in enumerate(y_recv_p)]
        y_recv_n = np.matmul(zs_n, f_locs)
        y_recv_n = [j*data_std[i]+data_mean[i] for i, j in enumerate(y_recv_n)]
        y_recv = np.matmul(zs, f_locs)
        y_recv = [j*data_std[i]+data_mean[i] for i, j in enumerate(y_recv)]
        
        y_pred_p = []
        y_pred_n = []
        y_pred = []
        for j in range(days, 0 , -1):
            ws_p = np.array([]).reshape(0, zs.shape[-1])
            ws_p = np.concatenate((ws_p, (dmfa.q_z_mu+dmfa.q_z_sig.exp())[-j,:max(dmfa.L)].detach().numpy()), axis = 0) #added
            ws_n = np.array([]).reshape(0, zs.shape[-1])
            ws_n = np.concatenate((ws_n, (dmfa.q_z_mu-dmfa.q_z_sig.exp())[-j,:max(dmfa.L)].detach().numpy()), axis = 0) #added
            ws = np.array([]).reshape(0, zs.shape[-1])
            ws = np.concatenate((ws, dmfa.q_z_mu[-j,:max(dmfa.L)].detach().numpy()), axis = 0) #added
            z_values = dmfa.q_z_mu[-j,:max(dmfa.L)].unsqueeze(0) #dmfa.q_z_0_mu[-j].unsqueeze(0) #edited
            z_t_1 = z_values.permute(1,0,2)[-np.array(dmfa.L)]
            z_t_1_s = z_values.permute(1,0,2)[-1]
            s_t_1 = dmfa.q_s[-j,max(dmfa.L)-1].unsqueeze(0)#torch.nn.Softmax(dim=-1)(dmfa.q_s_0[-j]).unsqueeze(0)
            for i in range(max(dmfa.L), T, 1): #range(T): edited
                if u_vals is not None and max(dmfa.L) == 1:
                    u_t_1 = u_vals[-j:-j+1,i].unsqueeze(0)
                else:
                    u_t_1 = None
                p_s = dmfa.strans(s_t_1, z_t_1_s) # 1 * S
                p_z_mu, p_z_sig = dmfa.trans(z_t_1, u_t_1) # S*1*z_dim
                z_val_p = (p_s.reshape(-1, 1, 1) * (p_z_mu+1*p_z_sig.exp())).sum(dim=0)
                z_val_n = (p_s.reshape(-1, 1, 1) * (p_z_mu-1*p_z_sig.exp())).sum(dim=0)
                z_val = (p_s.reshape(-1, 1, 1) * p_z_mu).sum(dim=0)
                z_values = torch.cat((z_values, dmfa.q_z_mu[-j,i].reshape(1,1,-1)), dim = 1)
                z_t_1 = z_values.permute(1,0,2)[-np.array(dmfa.L)]
                z_t_1_s = z_values.permute(1,0,2)[-1]
                ws_p = np.concatenate((ws_p, z_val_p.detach().numpy()), axis = 0)
                ws_n = np.concatenate((ws_n, z_val_n.detach().numpy()), axis = 0)
                ws = np.concatenate((ws, z_val.detach().numpy()), axis = 0)
                s_t_1 = dmfa.q_s[-j, i].unsqueeze(0)       
            y_pred_p.append(np.matmul(ws_p, f_locs)*data_std[-j]+data_mean[-j])
            y_pred_n.append(np.matmul(ws_n, f_locs)*data_std[-j]+data_mean[-j])
            y_pred.append(np.matmul(ws, f_locs)*data_std[-j]+data_mean[-j])
        y_pred_p = np.asarray(y_pred_p)
        y_pred_n = np.asarray(y_pred_n)
        y_pred = np.asarray(y_pred)
        
        for j in range(days, 0 , -1):
            for idx_loc in range(0, D, D//5):
                fig = plt.figure(figsize=(10,7))
                for i in range(I):
                    ax = fig.add_subplot(I,1,i+1)
                    ax.plot(dataa[-j].reshape(-1, D, I)[:,idx_loc, i], label = "Actual")
                    ax.plot(y_recv[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i], 'g-',
                            label = "Recovered", alpha = 0.8)
                    y_preds_p = y_pred_p[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i] * 1.0
                    y_preds_n = y_pred_n[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i] * 1.0
                    y_preds = y_pred[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i] * 1.0
                    ax.plot(y_preds, 'r-.', label = "Predicted", alpha = 0.8)
                    ax.fill_between(np.arange(len(y_preds)), y_preds_n, y_preds_p, color = 'red', alpha=0.1)
                    #ax.fill_between(np.arange(len(y_recv[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i])),
                    #                y_recv_n[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i],
                    #                y_recv_p[-j].reshape(-1, D, I)[:idxs[-j][0],idx_loc, i],
                    #                color = 'green', alpha=0.1)
                    ax.legend(framealpha = 0, fontsize=13)
                    if ID is not None:
                        ax.set_title(titles[data_id]+' #%d'%idx_loc, fontsize=fontsize)
                        ax.set_ylabel(ylabels[data_id][i], fontsize=fontsize+2)
                        ax.set_xlabel('Time -%d'%j, fontsize=fontsize+2)
                plt.tight_layout()
                fig.savefig(fig_PATH + "%s(-%d)prediction_roll_short%s" %(prefix,j,idx_loc) + ext, bbox_inches='tight')
            plt.close('all')
        
        RMSE = [np.power(dataa[i][idxs[i][1]] - y_pred[i,:idxs[i][0]][idxs[i][1]],2) for i in range(-days,0)]
        RMSE = np.sqrt(sum([i.sum() for i in RMSE])/sum([len(i) for i in RMSE]))
        print('Prediction RMSE %.2f' %RMSE)
        MAPE = [np.absolute((dataa[i][idxs[i][1]] - y_pred[i,:idxs[i][0]][idxs[i][1]])/dataa[i][idxs[i][1]]) for i in range(-days,0)]
        MAPE = sum([i.sum() for i in MAPE])/sum([len(i) for i in MAPE])*100
        print('Prediction MAPE %.2f' %MAPE)
        NRMSE = [dataa[i][idxs[i][1]]**2 for i in range(-days,0)]
        NRMSE = RMSE/np.sqrt(sum([i.sum() for i in NRMSE])/sum([len(i) for i in NRMSE]))*100
        print('Prediction NRMSE %.2f' %NRMSE)

        # plot spatial components
        factor_dim = len(dmfa.q_F_loc_mu)
        factors = dmfa.q_F_loc_mu.detach().numpy().reshape(factor_dim, -1, I)
        for k in np.arange(factor_dim)[:5]: # 5 components at most
            fig = plt.figure(figsize=(15,10))
            for i in range(I):
                ax = fig.add_subplot(I,1,i+1)
                ax.stem(factors[k,:,i], markerfmt=' ', use_line_collection = True)
                ax.set_title('factor'+' #%d'%(k+1), fontsize=fontsize)
                if ID is not None:
                    ax.set_ylabel(ylabels[data_id][i], fontsize=fontsize+2)
                    ax.set_xlabel(titles[data_id], fontsize=fontsize+2)
                if spat is not None:
                    ax.set_xticks(np.arange(D))
                    ax.set_xticklabels(spat)
            plt.tight_layout()
            fig.savefig(fig_PATH + "%sfactor%d" %(prefix,k) + ext, bbox_inches='tight')
        plt.close('all')
        
        # plot factors correlation
        factors = factors.reshape(factor_dim, -1)
        print('factors corrcoef: ', np.corrcoef(factors)[np.triu_indices(factor_dim,1)])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mesh = ax.matshow(np.corrcoef(factors))
        plt.colorbar(mesh, ax=ax)
        fig.savefig(fig_PATH + "%sfactors_corrcoef" %prefix + ext, bbox_inches='tight')
        