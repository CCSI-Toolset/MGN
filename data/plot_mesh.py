import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
from sqlalchemy import true

def get_comsol_data(fn = 'cylinder_flow_comsol.csv'):
    '''
    Preprocesses COMSOL cylinder flow simulation output.

    '''

    D = pd.read_csv(fn)
    x = D['x']
    y = D['y']
    D = D.drop(columns=['x','y'])

    X = D.values

    inds = np.arange(0,X.shape[1],4)
    times = X[:,inds]
    t = times[0]

    inds = np.arange(1,X.shape[1],4)
    vel_x = X[:,inds]

    inds = np.arange(2,X.shape[1],4)
    vel_y = X[:,inds]

    inds = np.arange(3,X.shape[1],4)
    p = X[:,inds]

    return x, y, t, vel_x, vel_y, p

def get_comsol_mesh(mesh_file = 'mesh_comsol_output.txt'):

    '''
    Preprocesses COMSOL cylinder flow mesh

    This function is necessary because the node coordinates and comsol mesh are in a different order
    Need to re-order the edge list from the mesh file
    '''

    def splitFloatLine(line):
        return list(map(float, line.split()[:2]))

    def splitElementLine(line):
        return list(map(int,line.split()[:3]))

    def simplexToEdgeList(simp):
        edges = [(simp[0], simp[1]), (simp[1], simp[2]), (simp[2], simp[0])]
        r_edges = [(e[1],e[0]) for e in edges]
        return edges + r_edges

    with open(mesh_file) as fid:
        mesh = fid.readlines()

    #get nodes
    nodeLine = mesh[4]
    numNodes = int(nodeLine.split()[2])
    mesh_nodes = mesh[10:(10+numNodes)]
    mesh_nodes = np.array(list(map(splitFloatLine, mesh_nodes)))

    #get mesh elements
    mesh_elements = mesh[11+numNodes:]
    mesh_elements = np.array(list(map(splitElementLine, mesh_elements)))
    mesh_elements = mesh_elements - 1 # comsol starts from 1 not 0.

    # #match mesh and node coordinates
    # Y = cdist(mesh_nodes, node_coordinates)
    # index = np.argmin(Y, axis=1)
    # simplex = index[mesh_elements]
    
    # A = list(map(simplexToEdgeList, simplex))
    # edge_list = [b for sublist in A for b in sublist]
    # edge_list = np.unique(edge_list,axis=1)
    
    return mesh_nodes, mesh_elements

x, y, t, vel_x, vel_y, p = get_comsol_data()
node_coordinates = np.vstack([x,y]).T

plt.figure(clear=True)
plt.plot(x,y,'o',color='blue',label='blue')
plt.plot(node_coordinates[:,0],node_coordinates[:,1],'.',color='red',label='mesh coordinates')
plt.legend()
plt.show()

print('hello')