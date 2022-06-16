import time
from os.path import exists

import igraph
import numpy as np
import openpyxl
import pandas as pd


def calc_inversematrix(pr, graph):
    # Calculate D
    d_vec = graph.degree(mode="all")
    d = np.reshape(np.repeat(d_vec, len(d_vec)), (len(d_vec), len(d_vec)))
    # Calculate A
    a = graph.get_adjacency()
    # Calculate W
    w = np.array(a.data) / d  # convert igraph matrix to numpy array
    # Calculate I
    identity = np.identity(w.shape[0])
    # Calculate inv_denom
    inv_matrix = np.linalg.inv(identity - (1 - pr) * w)
    return inv_matrix


def get_base_matrix(graph, nodes2label, nodes2labelvalues):
    # Get list of nodes in network
    nodes_network = graph.vs()["name"]

    # Creating x0
    x0 = [0] * len(nodes_network)
    for iterator, node in enumerate(nodes2label):
        # if node in nodes_network:
        network_index = nodes_network.index(node)
        x0[network_index] = nodes2labelvalues[iterator]

    return np.array(x0)


def run_propagate(pr, graph, inv_matrix, nodes2label, nodes2labelvalues, flag_RemoveSelfHeat=True):
    x0 = get_base_matrix(graph, nodes2label, nodes2labelvalues)
    # Match your list to network list of nodes
    # inds_network = []
    # for node in nodes2label:
    #     if node in nodes_network:
    #         inds_network.append(nodes_network.index(node))
    #     else:
    #         inds_network.append(-1)

    # Perform Network Propagation
    print('Performing network propagation ...')
    xss_t = np.matmul((x0 * pr), inv_matrix)

    # Remove self-heat
    if flag_RemoveSelfHeat:
        inv_denom_self_heat = np.diagonal(inv_matrix)
        xss_t = xss_t - (x0 * inv_denom_self_heat * pr)

    print('Done')
    return xss_t, x0


def run_propagate_perm(pr, graph, inv_denom, nodes2label, nodes2labelvalues, num_perms, flag_RemoveSelfHeat=True,
                       flag_PermuteInLabeled=False):
    x0 = get_base_matrix(graph, nodes2label, nodes2labelvalues)
    # Creating Randomly Permuted Matrix
    print('Creating Randomly Permuted Matrix ...')
    x0_perm_mat = np.zeros((num_perms, len(x0)))
    if flag_PermuteInLabeled:
        for i in range(num_perms):
            new_labels = np.random.permutation(nodes2labelvalues)
            x0_row = get_base_matrix(graph, nodes2label, new_labels)
            x0_perm_mat[i, :] = x0_row
    else:
        for i in range(num_perms):
            x0_perm_mat[i, :] = np.random.permutation(x0)

    # Run Network Propagation
    print('Running Network Propagation ...')
    xss_perm_mat = np.matmul(x0_perm_mat * pr, inv_denom)

    # Remove self-heat
    if flag_RemoveSelfHeat:
        inv_denom_self_heat = np.diagonal(inv_denom)
        xss_perm_mat = xss_perm_mat - np.multiply(x0_perm_mat, np.tile(inv_denom_self_heat, (num_perms, 1))) * pr

    return xss_perm_mat


def main():
    # ------------------------------------------------------------------------------------------------------------------------
    # get gene list
    print("reading excel file")
    excel_file = "Seven_CoV_MAGeCK_out_dummy_collapsed.xlsx"
    wb = openpyxl.load_workbook(excel_file)
    sheets = wb.sheetnames

    tmp_df = pd.read_csv("PathwayCommons12_Andreas_BaseNetwork_edges.txt", header=0)
    network_genes = pd.unique(pd.concat([tmp_df.iloc[:, 0], tmp_df.iloc[:, 1]]))

    # read gene summary table from MAGeCK
    dfs = {}
    for sheet in sheets:
        if sheet != "ReactomeFI":
            # print(sheet)
            df = pd.read_excel(excel_file, sheet_name=sheet)
            # filtering out nodes that are not in graph
            idx_to_keep = df.id.isin(network_genes)
            df = df.loc[idx_to_keep, :]
            dfs[sheet] = df

    # Load Gene List
    nodes2label = {}
    nodes2labelvalues = {}

    for sheet in sheets:
        print("Loading", sheet)
        nodes2label[sheet] = dfs[sheet].id.to_numpy()
        nodes2labelvalues[sheet] = -np.log10(dfs[sheet]['pos|score']).to_numpy()

    # ------------------------------------------------------------------------------------------------------------------------
    # Load Graph (Network)
    print("Load Graph (Network)")

    G_network = pd.read_csv("PathwayCommons12_Andreas_BaseNetwork_edges.txt", header=0)
    G_network = G_network.iloc[:, 0:2]
    g = igraph.Graph.DataFrame(G_network, directed=False)
    g = g.decompose(maxcompno=2)
    pr = 0.2

    if exists("PC_medhi_inv_denom_nodir.npy"):
        print("Found precomputed inv_denom in PC_medhi_inv_denom_nodir.RDS, loading it from file")
        inv_denom = np.load("PC_medhi_inv_denom_nodir.npy")
    else:  # calculate inv denom and save it to file
        print("calculating Inverse Matrix..")
        start = time.process_time()
        inv_denom = calc_inversematrix(pr, g[0])
        print("Done in:", time.process_time() - start)
        np.save("PC_medhi_inv_denom_nodir", inv_denom)

    # ------------------------------------------------------------------------------------------------------------------------
    # run propagation, permutation and calculate p-values

    xss_t_lst = {}
    xss_perm_mat_lst = {}
    if exists("xss_t_lst.npy"):
        xss_t_lst = np.load("xss_t_lst.npy", allow_pickle=True)
    if exists("xss_perm_mat_lst.npy"):
        xss_perm_mat_lst = np.load("xss_perm_mat_lst.npy", allow_pickle=True)

    # start propagation
    for i in nodes2label.keys():
        if i in xss_t_lst:
            print(i, " found in loaded object, skipping")
        else:
            print("processing ", i)
            # propagation
            print("running propagation...")
            start = time.process_time()
            xss_t, _ = run_propagate(pr, g[0], inv_denom, nodes2label=nodes2label[i],
                                      nodes2labelvalues=nodes2labelvalues[i], flag_RemoveSelfHeat=False)
            print("Done in:", time.process_time() - start)

            # permutation
            print("running permutation...")
            start = time.process_time()
            num_perms = 20000
            xss_perm_mat = run_propagate_perm(pr, g[0], inv_denom, nodes2label[i], nodes2labelvalues[i], num_perms,
                                              flag_RemoveSelfHeat=False,
                                              flag_PermuteInLabeled=False)
            print("Done in:", time.process_time() - start)

            xss_t_lst[i] = xss_t
            xss_perm_mat_lst[i] = xss_perm_mat
            ###

    np.save("xss_t_lst", xss_t_lst)
    np.save("xss_perm_mat_lst", xss_perm_mat_lst)

    # write propagated pvals for each screen
    # for i in nodes2label.keys():
    #     if i in xss_t_lst:



if __name__ == "__main__":
    main()
