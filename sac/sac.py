

import numpy as np
import networkit as nk
import torch
import torch.nn.functional as F

nk.setLogLevel("ERROR")

def batch_knn_graph_gpu(embeddings, k, batch_size=1024, device="cpu"):
    n, d = embeddings.shape
    st = 0
    k = int(k)

    graph_nk = nk.Graph(n, weighted=True, directed=False)
    for st in range(0, n, batch_size):
        ed = min(n, st+batch_size)
        sub_emb = embeddings[st:ed].to(device)
        with torch.no_grad():
            sub_sim = torch.mm(sub_emb, embeddings.T)
            values, indices = sub_sim.topk(k, dim=-1, sorted=False)
            # values = F.relu(values)
            values = values.to("cpu")
            indices = indices.to("cpu")

        sub_row = np.repeat(np.arange(st,ed), k)
        sub_col = indices.flatten()
        sub_data = values.flatten()

        sub_row = np.array(sub_row)
        sub_col = np.array(sub_col)
        sub_data = np.array(sub_data)

        mask = np.where(sub_col >= sub_row)[0] ## Avoid duplicated insert
        sub_row = sub_row[mask]
        sub_col = sub_col[mask]
        sub_data = sub_data[mask]

        graph_nk.addWeightedEdges(sub_row, sub_col, sub_data, checkMultiEdge=False)

    return graph_nk


def batch_delta_knn_graph_gpu(embeddings, k1, k2, batch_size=1024, device="cpu"):
    n, d = embeddings.shape
    st = 0
    k1 = int(k1)
    k2 = int(k2)
    assert k2 > k1, f"Invalid k1&k2, k2 ({k2}) should be larger than k1 ({k1})"

    graph_nk = nk.Graph(n, weighted=True, directed=False)
    for st in range(0, n, batch_size):
        ed = min(n, st+batch_size)
        sub_emb = embeddings[st:ed].to(device)
        with torch.no_grad():
            sub_sim = torch.mm(sub_emb, embeddings.T)
            values1, indices1 = sub_sim.topk(k1, dim=-1, sorted=False)
            sub_row = np.repeat(np.arange(0, ed-st), k1)
            sub_col = indices1.flatten()
            sub_sim[sub_row, sub_col] = -2

            values, indices = sub_sim.topk(k2-k1, dim=-1, sorted=False)
            
            values = values.to("cpu")            
            indices = indices.to("cpu")


        sub_row = np.repeat(np.arange(st,ed), k2-k1)
        sub_col = indices.flatten()
        sub_data = values.flatten()

        sub_row = np.array(sub_row)
        sub_col = np.array(sub_col)
        sub_data = np.array(sub_data)

        mask = np.where(sub_col >= sub_row)[0] ## Avoid duplicated insert
        sub_row = sub_row[mask]
        sub_col = sub_col[mask]
        sub_data = sub_data[mask]

        graph_nk.addWeightedEdges(sub_row, sub_col, sub_data, checkMultiEdge=False)

    return graph_nk


def sac(embeddings, m, batch_size=1024, maxIter=10, neighbor_mode="queue", device="cpu"):
    embeddings = torch.FloatTensor(embeddings).to(device)
    embeddings = F.normalize(embeddings, dim=1, p=2)
    n, d = embeddings.shape
    sigmamax = int(n*n/m)
    sigma0 = 10**(int(np.log10(sigmamax))-1) #### 695-->10, 4385-->100
    sigma0 = min(sigma0, 100) ### Too large sigma0 will downgrade the efficiency

    #### Initial kNN graph
    k0 = np.ceil(sigma0 * m / n)
    graph0_nk = batch_knn_graph_gpu(embeddings, k0, batch_size, device)
    algo = nk.community.PLM(graph0_nk, refine=True, turbo=True, nm=neighbor_mode, par="balanced", maxIter=32)
    zeta0 = nk.community.detectCommunities(graph0_nk, algo=algo, inspect=False)
    coarsened_graph0_nk, fine_to_coarsen0 = algo.coarsen(graph0_nk, zeta0)

    #### Progressive Graph Construction
    for d_idx, sigma in enumerate(np.arange(2*sigma0, sigmamax, sigma0)):
        if d_idx == maxIter-1:
            print(f"Reach maxiter ({maxIter}). Break at {d_idx+1}th graph, sigma= {sigma}")
            break

        k = np.ceil(sigma * m / n)
        last_k = np.ceil((sigma-sigma0) * m / n)

        ##### delta kNN graph construction
        delta_graph_nk = batch_delta_knn_graph_gpu(embeddings, last_k, k, batch_size, device)

    
        algo = nk.community.PLM(delta_graph_nk, refine=True, turbo=True, nm=neighbor_mode, par="balanced", maxIter=32)
        coarsened_graph0_nk = algo.progressiveOnline(delta_graph_nk, coarsened_graph0_nk, zeta0)

        if algo.stopNow() == True:
            print(f"Break at {d_idx+1}th graph, sigma= {sigma}")
            break
        else:
            print(f"Accept {d_idx+1}th graph, sigma= {sigma}")

    algo_final = nk.community.PLM(coarsened_graph0_nk, refine=True, turbo=True, nm=neighbor_mode, par="balanced", maxIter=32)
    coarsen_zeta = nk.community.detectCommunities(coarsened_graph0_nk, algo=algo_final, inspect=False)
    zeta = algo_final.prolong(coarsened_graph0_nk, coarsen_zeta, graph0_nk, fine_to_coarsen0)
    preds = zeta.getVector()

    return preds


def sac_wo_hierarchy(embeddings, m, batch_size=1024, maxIter=10, neighbor_mode="queue", device="cpu", return_graph=False):
    ### m = 2 * #Undirected Edges in attributed graph
    embeddings = torch.FloatTensor(embeddings).to(device)
    embeddings = F.normalize(embeddings, dim=1, p=2)
    n, d = embeddings.shape
    sigmamax = int(n*n/m)
    sigma0 = 10**(int(np.log10(sigmamax))-1) #### 695-->10, 4385-->100
    sigma0 = min(sigma0, 100) ### Too large sigma0 will downgrade the efficiency

    #### Initial kNN graph
    k0 = np.ceil(sigma0 * m / n)
    graph0_nk = batch_knn_graph_gpu(embeddings, k0, batch_size, device)

    algo = nk.community.PLM(graph0_nk, refine=True, turbo=True, nm=neighbor_mode, par="balanced", maxIter=32)
    zeta0 = nk.community.detectCommunities(graph0_nk, algo=algo, inspect=False)


    #### Progressive Graph Construction
    for d_idx, sigma in enumerate(np.arange(2*sigma0, sigmamax, sigma0)):
        if d_idx == maxIter-1:
            print(f"Reach maxiter ({maxIter}). Break at {d_idx+1}th graph, sigma= {sigma}")
            break

        k = np.ceil(sigma * m / n)
        last_k = np.ceil((sigma-sigma0) * m / n)

        ##### delta kNN graph construction
        delta_graph_nk = batch_delta_knn_graph_gpu(embeddings, last_k, k, batch_size, device)

        algo = nk.community.PLM(delta_graph_nk, refine=True, turbo=True, nm=neighbor_mode, par="balanced", maxIter=32)
        graph0_nk = algo.progressiveOnline_wo_hierarchy(delta_graph_nk, graph0_nk, zeta0)

        if algo.stopNow() == True:
            print(f"Break at {d_idx+1}th graph, sigma= {sigma}")
            break
        else:
            print(f"Accept {d_idx+1}th graph, sigma= {sigma}")

    algo_final = nk.community.PLM(graph0_nk, refine=True, turbo=True, nm=neighbor_mode, par="balanced", maxIter=32)
    zeta = nk.community.detectCommunities(graph0_nk, algo=algo_final, inspect=False)
    preds = np.array(zeta.getVector())

    if return_graph:
        return preds, graph0_nk
    else:
        return preds

