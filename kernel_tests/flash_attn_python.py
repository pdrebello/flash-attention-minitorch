import torch 
import numpy as np

def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1).values[:, None])
    return e_x / torch.sum(e_x, dim=-1)[:, None]

def compute_attention(Q, K, V):
    N, d = Q.shape
    tau = np.sqrt(1.0/d)
    attention_scores = Q @ K.T
    attention_weights = softmax((attention_scores) * tau)
    O = torch.matmul(attention_weights, V)
    return O

def flash_attention(Q, K, V): 
    
    N, d = Q.shape
    tau = np.sqrt(1.0/d)
    #on_chip_memory_size = d * 256
    #B_c = on_chip_memory_size // (4 * d)  # Using 4 bytes per float
    #B_r = min(on_chip_memory_size // (4 * d), d)
    #B_c = 32
    #B_r = min(B_c, d)

    B_c = 16
    B_r = min(B_c, d)

    O = torch.zeros_like(Q, device=Q.device)
    l = torch.zeros(N, dtype=torch.float64, device=Q.device)
    m = -np.inf * torch.ones(N, dtype=torch.float64, device=Q.device)
    T_r = int(np.ceil(N / B_r))
    T_c = int(np.ceil(N / B_c))
    for j in range(T_c):
        K_j = K[j * B_c:(j + 1) * B_c]
        V_j = V[j * B_c:(j + 1) * B_c]
        for i in range(T_r):
            Q_i = Q[i * B_r:(i + 1) * B_r]
            
            O_i = O[i * B_r:(i + 1) * B_r]
            l_i = l[i * B_r:(i + 1) * B_r]
            m_i = m[i * B_r:(i + 1) * B_r]
    
            S_ij = tau * (Q_i @ K_j.T) 
            m_ij = torch.max(S_ij, dim=1).values
            P_ij = torch.exp(S_ij - m_ij[:, None]) 
            l_ij = torch.sum(P_ij, dim=1)
            m_new = torch.maximum(m_i, m_ij) 
            
            l_new = torch.exp(m_i - m_new) * l_i + torch.exp(m_ij - m_new) * l_ij
            O[i * B_r:(i + 1) * B_r] = (1.0/l_new)[:, None] * ((torch.exp(m_i - m_new) * l_i)[:, None] * O_i +  (torch.exp(m_ij - m_new)[:, None] * (P_ij @ V_j)))

            m[i * B_r:(i + 1) * B_r] = m_new
            
            l[i * B_r:(i + 1) * B_r] = l_new
    return O, l, m


def flash_attention2(Q, K, V): 

    N, d = Q.shape
    tau = np.sqrt(1.0/d)
    B_c = 4
    B_r = min(B_c, d)

    O = torch.zeros_like(Q, device=Q.device)
    L = torch.zeros(N, dtype=torch.float64, device=Q.device)
    #m = -np.inf * torch.ones(N, dtype=torch.float64, device=Q.device)
    T_r = int(np.ceil(N / B_r))
    T_c = int(np.ceil(N / B_c))
    for i in range(T_r):
        Q_i = Q[i * B_r:(i + 1) * B_r]   
        O_i = torch.zeros_like(Q_i, device=Q.device)

        l_i = torch.zeros(len(O_i), device=Q.device) #l[i * B_r:(i + 1) * B_r]
        m_i = torch.ones(len(O_i), device=Q.device) * (-np.inf) #m[i * B_r:(i + 1) * B_r]
        
        for j in range(T_c):
            K_j = K[j * B_c:(j + 1) * B_c]
            V_j = V[j * B_c:(j + 1) * B_c]
  
            S_ij = tau * (Q_i @ K_j.T) 
            m_i_prev = m_i.clone()
            m_i = torch.max(m_i, torch.max(S_ij, dim=1).values)
            P_ij = torch.exp(S_ij - m_i[:, None]) 
            l_i = torch.exp(m_i_prev - m_i)*l_i + torch.sum(P_ij, dim=1)
            
            if(j == 0):
                O_i = P_ij @ V_j
            else:
                O_i =  torch.diag(torch.exp(m_i_prev - m_i)) @ O_i + P_ij @ V_j #(1.0/torch.exp(m_i_prev - m_i))[:, None]

        O_i = torch.diag(1.0/l_i) @ O_i #(1.0/l_i)[:, None] * O_i
        L_i = m_i + torch.log(l_i)
        O[i * B_r:(i + 1) * B_r] = O_i
        L[i * B_r:(i + 1) * B_r] = L_i
    #print(L)
    return O, L

def flash_attention_backward(Q, K, V, O_flash, dO, l, m): 
    
    N, d = Q.shape
    tau = np.sqrt(1.0/d)
    on_chip_memory_size = d * 256
    B_c = on_chip_memory_size // (4 * d)  # Using 4 bytes per float
    B_r = min(on_chip_memory_size // (4 * d), d)
    B_c = 4
    B_r = min(B_c, d)
    
    dQ = torch.zeros_like(Q, device=Q.device)
    dK = torch.zeros_like(K, device=Q.device)
    dV = torch.zeros_like(V, device=Q.device)
    T_r = int(np.ceil(N / B_r))
    T_c = int(np.ceil(N / B_c))

    for j in range(T_c):
        K_j = K[j * B_c:(j + 1) * B_c]
        V_j = V[j * B_c:(j + 1) * B_c]
        dK_j = dK[j * B_c:(j + 1) * B_c]
        dV_j = dV[j * B_c:(j + 1) * B_c]
        
        for i in range(T_r):
            Q_i = Q[i * B_r:(i + 1) * B_r]
            O_i = O_flash[i * B_r:(i + 1) * B_r]
            dQ_i = dQ[i * B_r:(i + 1) * B_r]
            dO_i = dO[i * B_r:(i + 1) * B_r]
            l_i = l[i * B_r:(i + 1) * B_r]
            m_i = m[i * B_r:(i + 1) * B_r]
            
            S_ij = tau * (Q_i @ K_j.T) 
            P_ij = (1.0/l_i)[:, None] * torch.exp(S_ij - m_i[:, None])  
            
            dV_j = dV_j + (P_ij.T @ dO_i)
            dP_ij = dO_i @ V_j.T
            D_i = (dO_i * O_i).sum(dim=1)
            
 
            dS_ij = P_ij * (dP_ij - D_i[:, None])
            dQ[i * B_r:(i + 1) * B_r] = dQ_i + (tau * dS_ij @ K_j)
            dK_j = dK_j + tau * dS_ij.T @ Q_i

            
        dK[j * B_c:(j + 1) * B_c] = dK_j
        dV[j * B_c:(j + 1) * B_c] = dV_j
    return dQ, dK, dV

def flash_attention2_backward(Q, K, V, O_flash, dO, L): 

    N, d = Q.shape
    tau = np.sqrt(1.0/d)
    on_chip_memory_size = d * 256
    B_c = on_chip_memory_size // (4 * d)  # Using 4 bytes per float
    B_r = min(on_chip_memory_size // (4 * d), d)
    B_c = 4
    B_r = min(B_c, d)
    
    dQ = torch.zeros_like(Q, device=Q.device)
    dK = torch.zeros_like(K, device=Q.device)
    dV = torch.zeros_like(V, device=Q.device)
    T_r = int(np.ceil(N / B_r))
    T_c = int(np.ceil(N / B_c))
    D = (dO * O_flash).sum(axis=1)
    for j in range(T_c):
        K_j = K[j * B_c:(j + 1) * B_c]
        V_j = V[j * B_c:(j + 1) * B_c]
        dK_j = dK[j * B_c:(j + 1) * B_c]
        dV_j = dV[j * B_c:(j + 1) * B_c]
        
        for i in range(T_r):
            Q_i = Q[i * B_r:(i + 1) * B_r]
            O_i = O_flash[i * B_r:(i + 1) * B_r]
            dQ_i = dQ[i * B_r:(i + 1) * B_r]
            dO_i = dO[i * B_r:(i + 1) * B_r]
            L_i = L[i * B_r:(i + 1) * B_r]
            D_i = D[i * B_r:(i + 1) * B_r]
            
            S_ij = tau * (Q_i @ K_j.T) 
            P_ij = torch.exp(S_ij - L_i[:, None])  #(1.0/l_i)[:, None] * 
            
            dV_j = dV_j + (P_ij.T @ dO_i)
            dP_ij = dO_i @ V_j.T

            D_i = (dO_i * O_i).sum(dim=1)
            dS_ij = P_ij * (dP_ij - D_i[:, None])
            
            dQ[i * B_r:(i + 1) * B_r] = dQ_i + (tau * dS_ij @ K_j)
            dK_j = dK_j + tau * dS_ij.T @ Q_i

            
        dK[j * B_c:(j + 1) * B_c] = dK_j
        dV[j * B_c:(j + 1) * B_c] = dV_j
    return dQ, dK, dV