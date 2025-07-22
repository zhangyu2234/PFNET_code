import torch
import torch.nn as nn
import torch.nn.functional as F
from dbgan_model import *
import pickle
import os
from MMD import distribution
from utils import _gradient_penalty
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import dense_to_sparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load v_obs, A_obs
train_vobs = os.path.join('./data_process', 'train_vobs.pkl')
train_Aobs = os.path.join('./data_process', 'train_Aobs.pkl')

# val
val_vobs = os.path.join('./data_process', 'val_vobs.pkl')
val_Aobs = os.path.join('./data_process', 'val_Aobs.pkl')

# test
test_vobs = os.path.join('./data_process', 'test_vobs.pkl')
test_Aobs = os.path.join('./data_process', 'test_Aobs.pkl')


""" 
# train
with open(train_vobs, 'rb') as f:
    v_obs_lst = pickle.load(f)

with open(train_Aobs, 'rb') as f:
    A_obs_lst = pickle.load(f)

save_path = './dbgan_embd'

for i in range(len(v_obs_lst)):
    v_obs_lst[i] = v_obs_lst[i].to(device)

for i in range(len(A_obs_lst)):
    A_obs_lst[i] = A_obs_lst[i].to(device)
# """

# val
#""" 
with open(val_vobs, 'rb') as f:
    v_obs_lst = pickle.load(f)

with open(val_Aobs, 'rb') as f:
    A_obs_lst = pickle.load(f)

save_path = './dbgan_embd_val'

for i in range(len(v_obs_lst)):
    v_obs_lst[i] = v_obs_lst[i].to(device)

for i in range(len(A_obs_lst)):
    A_obs_lst[i] = A_obs_lst[i].to(device)
#"""

# test
""" 
with open(test_vobs, 'rb') as f:
    v_obs_lst = pickle.load(f)

with open(test_Aobs, 'rb') as f:
    A_obs_lst = pickle.load(f)


save_path = './dbgan_embd_test'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(len(v_obs_lst)):
    v_obs_lst[i] = v_obs_lst[i].to(device)

for i in range(len(A_obs_lst)):
    A_obs_lst[i] = A_obs_lst[i].to(device)
"""



# parameters
in_features = 2  #(x, y)
hidden1 = 5
hidden2 = 2
hidden3 = 5

# DPP

def train(feas, adj, device):
    """ 
    params:
    feas: V_obs -> (N, feat=2)
    adj: A_obs -> (N, N)
    """

    # DPP Sample
    kde = distribution(adj, feas, sample_num=4)

    # model
    discriminator = Discriminator(in_dim=2, hidden1=hidden1, hidden3=hidden3).to(device)
    D_Graph = D_graph(in_features, hidden2).to(device)
    model = Encoder(in_features, hidden1, hidden2, dropout=0.2, num_nodes=feas.shape[0], num_layers=2).to(device)
    model_z2g = Generator_z2g(in_features, hidden1, hidden2, dropout=0.2, num_nodes=feas.shape[0], num_layers=2).to(device)

    # Optimizer
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.99))
    discriminator_optimizer_z2g = torch.optim.Adam(D_Graph.parameters(), lr=0.0001, betas=(0.9, 0.99))

    generator_optimizer_z2g = torch.optim.Adam(model_z2g.parameters(), lr=0.0001, betas=(0.9, 0.99))
    encoder_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99)) 

    # cal pos_weight, norm 
    pos_weight = float((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()) # 负样本的比重
    pos_weight = torch.Tensor([pos_weight]).to(device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    norm = torch.Tensor([norm]).to(device)

    # adj -> edge_index
    edge_index, _ = dense_to_sparse(adj)

    # DPP 
    z_real_dist = kde.sample(adj.shape[0]) 
    z_real_dist = torch.FloatTensor(z_real_dist).to(device)
    

    for _ in range(5):
    
        # D_graph == Dx
        discriminator_optimizer_z2g.zero_grad()
        real_data = feas
        GD_real = D_Graph(feas)
        z2g = model_z2g(z_real_dist, edge_index)
        GD_fake = D_Graph(z2g.detach())
        generated_data = z2g
        gradient_penalty = _gradient_penalty(real_data, generated_data, D_Graph, device=device)
        GD_loss = GD_fake.mean() - GD_real.mean() + gradient_penalty
        GD_loss.backward()
        discriminator_optimizer_z2g.step()

    # Generator
    generator_optimizer_z2g.zero_grad()
    z2g = model_z2g(z_real_dist, edge_index)
    GD_fake = D_Graph(z2g) # 2708, 1
   
    generator_loss_z2g = -GD_fake.mean()
    generator_loss_z2g.backward()
    generator_optimizer_z2g.step()

   
    # Discriminator == Dz
    discriminator_optimizer.zero_grad()
    real_data = z_real_dist
    generated_data, _ = model(feas, edge_index)
    gradient_penalty = _gradient_penalty(real_data, generated_data, discriminator, device=device)

    d_fake = discriminator(generated_data.detach())
    d_real = discriminator(real_data)
    discriminator_loss = d_fake.mean() - d_real.mean() + gradient_penalty
    discriminator_loss.backward()
    discriminator_optimizer.step()


    # Encoder
    encoder_optimizer.zero_grad()
    embeddings, preds_sub = model(feas, edge_index)
    labels_sub = adj.reshape(-1)

    cost = norm * F.binary_cross_entropy_with_logits(preds_sub, labels_sub, pos_weight)

    preds_cycle = model_z2g(embeddings, edge_index)
    labels_cycle = feas
    
    cost_cycle = norm * F.binary_cross_entropy_with_logits(preds_cycle, labels_cycle)
    reconstruction_loss = 0.01 * cost + cost_cycle # Reconstruction loss
    
    latent_dim, preds_sub = model(feas, edge_index)
    d_fake = discriminator(latent_dim)
    encoder_A_loss = -d_fake.mean()

    encoder_total_loss = encoder_A_loss + 0.01 * reconstruction_loss
    
    encoder_total_loss.backward()
    encoder_optimizer.step()


    all_loss = [encoder_total_loss.item(), generator_loss_z2g.item(),  GD_loss.item(), discriminator_loss.item()]
    emb = model.embedding

    return all_loss, emb


if __name__ == '__main__':
    num_epochs = 100
    emb_lst_total = []
    encoder_loss_lst_total = []
    generator_loss_lst_total = []
    GD_loss_lst_total = []
    disc_loss_lst_total = []

    for i, data in enumerate(zip(v_obs_lst, A_obs_lst)):
        feas, adj = data[0], data[1]
        encoder_loss_lst = []
        generator_loss_lst = []
        GD_loss_lst = []
        disc_loss_lst = []

        for epoch in range(num_epochs):
            all_loss, emb = train(feas, adj, device)

            encoder_loss_lst.append(all_loss[0])
            generator_loss_lst.append(all_loss[1])
            GD_loss_lst.append(all_loss[2])
            disc_loss_lst.append(all_loss[3])

            # embedding
            last_embeddings = [emb.detach().cpu()]

        # dataset total loss, embedding
        encoder_loss_lst_total.append(encoder_loss_lst)
        generator_loss_lst_total.append(generator_loss_lst)
        GD_loss_lst_total.append(GD_loss_lst)
        disc_loss_lst_total.append(disc_loss_lst)
        emb_lst_total.append(last_embeddings)
        
        if (i+1) % 10 == 0:
            print('dataset: {}, encoder_total_loss: {:.4f}, generator_loss_z2g: {:.4f},  GD_loss: {:.4f}, discriminator_loss: {:.4f}'.format(i+1, 
                                                                                                                                                 all_loss[0], 
                                                                                                                                                 all_loss[1], 
                                                                                                                                                 all_loss[2], 
                                                                                                                                                 all_loss[3]))
            
        
        
    
    data_dict = {'emb': emb_lst_total}
    emb_data = os.path.join(save_path, 'emb_lst.pkl')

    with open(emb_data, 'wb') as f:
        pickle.dump(data_dict, f)
    f.close() 
 


    
    """ 
    epochs = np.arange(100)
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, np.array(encoder_loss_lst), label='Encoder Loss')
    plt.plot(epochs, np.array(generator_loss_lst), label='Generator Loss')
    plt.plot(epochs, np.array(GD_loss_lst), label="GD Loss")
    plt.plot(epochs, np.array(disc_loss_lst), label="Discriminator Loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of Traning')
    plt.legend()
    plt.show()

    """





        


    





 




    

        
    

