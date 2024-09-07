"""
reference:
[1] https://arxiv.org/abs/2205.14135
"""
import numpy as np
import pytest


def softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    x = x - max_x
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


# standard attention
def standard_attention(Q, K, V):
    S = Q @ K.T
    P = softmax(S)
    O = P @ V
    return O


# flash attention
def flash_attention(Q, K, V, N, d, M):
    B_c = M // (4*d)
    B_r = min(B_c, d)
    # B_c = N // 2
    # B_r = N // 2

    O = np.zeros((N, d))
    l = np.zeros((N,))
    m = np.full((N,), -np.inf)
    T_r = N // B_r
    T_c = N // B_c

    for j in range(int(T_c)):
        # load K, V from HBM to on-chip SRAM
        K_j = K[j*B_c:(j+1)*B_c, :]  # (B_c, d)
        V_j = V[j*B_c:(j+1)*B_c, :]  # (B_c, d)
        for i in range(int(T_r)):
            Q_i = Q[i*B_r:(i+1)*B_r, :]  # (B_r, d)
            O_i = O[i*B_r:(i+1)*B_r, :]  # (B_r, d)
            l_i = l[i*B_r:(i+1)*B_r]  # (B_r,)
            m_i = m[i*B_r:(i+1)*B_r]  # (B_r,)

            S_ij = Q_i @ K_j.T  # (B_r, B_c)

            m_ij = np.max(S_ij, axis=1)  # (B_r,)
            P_ij = np.exp(S_ij - m_ij[:, None])  # (B_r, B_c)
            l_ij = np.sum(P_ij, axis=1)  # (B_r,)

            m_i_new = np.maximum(m_i, m_ij)  # (B_r,)
            l_i_new = np.exp(m_i - m_i_new) * l_i + np.exp(m_ij - m_i_new) * l_ij  # (B_r,)

            updated_old_O_i = l_i[:, None] * np.exp(m_i - m_i_new)[:, None] * O_i / l_i_new[:, None]
            new_O_i = np.exp(m_ij - m_i_new)[:, None] * P_ij @ V_j / l_i_new[:, None]
            sum_O_i = updated_old_O_i + new_O_i
            # O_i = (l_i[:, None] * np.exp(m_i - m_i_new)[:, None] * O_i + np.exp(m_ij - m_i_new)[:, None] * P_ij  @ V_j) / l_i_new[:, None]

            O[i*B_r:(i+1)*B_r, :] = sum_O_i

            l[i*B_r:(i+1)*B_r] = l_i_new
            m[i*B_r:(i+1)*B_r] = m_i_new
    return O


# pytest to compare the result of two attention
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("d", [256])
@pytest.mark.parametrize("M", [1024])
def test_attention(N, d, M):
    Q = np.random.randn(N, d)
    K = np.random.randn(N, d)
    V = np.random.randn(N, d)

    O1 = standard_attention(Q, K, V)
    O2 = flash_attention(Q, K, V, N, d, M)
    np.testing.assert_allclose(O1, O2, rtol=1e-05, atol=1e-05)


# construct some simple tests to debug
@ pytest.mark.parametrize("Q", [np.array([[0], [1]])])
@ pytest.mark.parametrize("K", [np.array([[0], [1]])])
@ pytest.mark.parametrize("V", [np.array([[0], [1]])])
def test_attention_1(Q, K, V):
    Q1 = standard_attention(Q, K, V)
    Q2 = flash_attention(Q, K, V, 2, 1, 2)

    np.testing.assert_allclose(Q1, Q2, rtol=1e-05, atol=1e-05)