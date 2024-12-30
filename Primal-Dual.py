import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

def primal_dual_afim_escala(A, b, c, x0, y0, z0, tau=0.9, max_iter=20, tol=1e-6):
    x, y, z = x0.copy(), y0.copy(), z0.copy()
    m, n = A.shape
    values = []
    y_values = []
    
    for k in range(max_iter):
        # Resíduos
        rp = b - A @ x
        rd = c - A.T @ y - z
        ra = -np.diag(x) @ np.diag(z) @ np.ones(n)
        primal_value = c.T @ x
        dual_value = b.T @ y
        values.append(primal_value)
        y_values.append(y.copy())

        # Critérios de parada
        primal_res = np.linalg.norm(rp) / (np.linalg.norm(b) + 1)
        dual_res = np.linalg.norm(rd) / (np.linalg.norm(c) + 1)
        gap_rel1 = np.abs(c.T @ x - b.T @ y) / (1 + np.abs(c.T @ x) + np.abs(b.T @ y))
        gap_rel2 = np.abs(x.T @ z) / (1 + np.abs(c.T @ x) + np.abs(b.T @ y))

        # Verificação dos critérios de parada
        if primal_res <= tol and dual_res <= tol and (gap_rel1 <= tol or gap_rel2 <= tol):
            print(f"Convergiu em {k+1} iterações!")
            break

        # Construção de D^-1 e M = AD^-1A^T
        D_inv = np.diag(1 / x) @ np.diag(z)
        M = A @ D_inv @ A.T

        # Decomposição de Cholesky de M
        L = cholesky(M, lower=True)

        # Resolver o sistema usando Cholesky
        rhs_dy = rp + A @ D_inv @ (rd - np.linalg.inv(np.diag(x)) @ ra)
        w = solve_triangular(L, rhs_dy, lower=True)
        dy = solve_triangular(L.T, w, lower=False)

        # Resolver para dx e dz
        dx = D_inv @ (A.T @ dy - rd + np.linalg.inv(np.diag(x)) @ ra)
        dz = np.linalg.inv(np.diag(x)) @ (ra - np.diag(z) @ dx)

        # Passos primal e dual
        rho_p = min([(-x[i] / dx[i]) for i in range(n) if dx[i] < 0], default=1)
        rho_d = min([(-z[i] / dz[i]) for i in range(n) if dz[i] < 0], default=1)
        alpha_p = min(1, tau * rho_p)
        alpha_d = min(1, tau * rho_d)

        # Atualização das variáveis
        x += alpha_p * dx
        y += alpha_d * dy
        z += alpha_d * dz

        # Progresso realizado
        print(f"Iteração {k+1}:")
        print(f"Solução primal (x): {x}, Valor primal: {primal_value:.4f}")
        print(f"Solução dual (y): {y}, Valor dual: {dual_value:.4f}\n")

    # Resultado final
    print("Solução final (Afim-Escala):")
    print(f"Solução primal (x): {x}, Valor primal: {primal_value:.4f}")
    print(f"Solução dual (y): {y}, Valor dual: {dual_value:.4f}\n")
    return x, y_values, z, values

def primal_dual_classico(A, b, c, x0, y0, z0, tau=0.9, sigma=0.1, max_iter=20, tol=1e-6):
    x, y, z = x0.copy(), y0.copy(), z0.copy()
    m, n = A.shape
    values = []
    y_values = []

    for k in range(max_iter):
        # Cálculo de mu
        mu = sigma * (x @ z) / n

        # Resíduos
        rp = b - A @ x
        rd = c - A.T @ y - z
        rc = mu * np.ones(n) - np.diag(x) @ np.diag(z) @ np.ones(n)
        primal_value = c.T @ x
        dual_value = b.T @ y
        values.append(primal_value)
        y_values.append(y.copy())

        # Critérios de parada
        primal_res = np.linalg.norm(rp) / (np.linalg.norm(b) + 1)
        dual_res = np.linalg.norm(rd) / (np.linalg.norm(c) + 1)
        gap_rel1 = np.abs(c.T @ x - b.T @ y) / (1 + np.abs(c.T @ x) + np.abs(b.T @ y))
        gap_rel2 = np.abs(x.T @ z) / (1 + np.abs(c.T @ x) + np.abs(b.T @ y))

        # Verificação dos critérios de parada
        if primal_res <= tol and dual_res <= tol and (gap_rel1 <= tol or gap_rel2 <= tol):
            print(f"Convergiu em {k+1} iterações!")
            break

        # Construção de D^-1 e M = AD^-1A^T
        D_inv = np.diag(1 / x) @ np.diag(z)
        M = A @ D_inv @ A.T

        # Decomposição de Cholesky de M
        L = cholesky(M, lower=True)

        # Resolver o sistema usando Cholesky
        rhs_dy = rp + A @ D_inv @ (rd - np.linalg.inv(np.diag(x)) @ rc)
        w = solve_triangular(L, rhs_dy, lower=True)
        dy = solve_triangular(L.T, w, lower=False)

        # Resolver para dx e dz
        dx = D_inv @ (A.T @ dy - rd + np.linalg.inv(np.diag(x)) @ rc)
        dz = np.linalg.inv(np.diag(x)) @ (rc - np.diag(z) @ dx)

        # Passos primal e dual
        rho_p = min([(-x[i] / dx[i]) for i in range(n) if dx[i] < 0], default=1)
        rho_d = min([(-z[i] / dz[i]) for i in range(n) if dz[i] < 0], default=1)
        alpha_p = min(1, tau * rho_p)
        alpha_d = min(1, tau * rho_d)

        # Atualização das variáveis
        x += alpha_p * dx
        y += alpha_d * dy
        z += alpha_d * dz

        # Progresso realizado
        print(f"Iteração {k+1}:")
        print(f"Solução primal (x): {x}, Valor primal: {primal_value:.4f}")
        print(f"Solução dual (y): {y}, Valor dual: {dual_value:.4f}\n")

    # Resultado final
    print("Solução final (Clássico):")
    print(f"Solução primal (x): {x}, Valor primal: {primal_value:.4f}")
    print(f"Solução dual (y): {y}, Valor dual: {dual_value:.4f}\n")
    return x, y_values, z, values

def primal_dual_preditor_corrector(A, b, c, x0, y0, z0, tau=0.9, max_iter=20, tol=1e-6):
    x, y, z = x0.copy(), y0.copy(), z0.copy()
    m, n = A.shape
    values = []
    y_values = []

    for k in range(max_iter):
        # Resíduos
        rp = b - A @ x
        rd = c - A.T @ y - z
        ra = -np.diag(x) @ np.diag(z) @ np.ones(n)
        primal_value = c.T @ x
        dual_value = b.T @ y
        values.append(primal_value)
        y_values.append(y.copy())

        # Critérios de parada
        primal_res = np.linalg.norm(rp) / (np.linalg.norm(b) + 1)
        dual_res = np.linalg.norm(rd) / (np.linalg.norm(c) + 1)
        gap_rel1 = np.abs(c.T @ x - b.T @ y) / (1 + np.abs(c.T @ x) + np.abs(b.T @ y))
        gap_rel2 = np.abs(x.T @ z) / (1 + np.abs(c.T @ x) + np.abs(b.T @ y))

        # Verificação dos critérios de parada
        if primal_res <= tol and dual_res <= tol and (gap_rel1 <= tol or gap_rel2 <= tol):
            print(f"Convergiu em {k+1} iterações!")
            break

        # Construção de D^-1 e M = AD^-1A^T
        D_inv = np.diag(1 / x) @ np.diag(z)
        M = A @ D_inv @ A.T

        # Decomposição de Cholesky de M
        L = cholesky(M, lower=True)

        # Resolver o sistema linear usando Cholesky
        rhs_dy = rp + A @ D_inv @ (rd - np.linalg.inv(np.diag(x)) @ ra)
        w = solve_triangular(L, rhs_dy, lower=True)
        dy = solve_triangular(L.T, w, lower=False)

        # Resolver dx e dz
        dx = D_inv @ (A.T @ dy - rd + np.linalg.inv(np.diag(x)) @ ra)
        dz = np.linalg.inv(np.diag(x)) @ (ra - np.diag(z) @ dx)

        # Passos primal e dual
        rho_p = min([(-x[i] / dx[i]) for i in range(n) if dx[i] < 0], default=1)
        rho_d = min([(-z[i] / dz[i]) for i in range(n) if dz[i] < 0], default=1)
        alpha_p = min(1, tau * rho_p)
        alpha_d = min(1, tau * rho_d)

        # Atualização das variáveis
        x += alpha_p * dx
        y += alpha_d * dy
        z += alpha_d * dz

        # Progresso realizado
        print(f"Iteração {k+1}:")
        print(f"Solução primal (x): {x}, Valor primal: {primal_value:.4f}")
        print(f"Solução dual (y): {y}, Valor dual: {dual_value:.4f}")
        print(f"Norma dos resíduos: Primal {primal_res:.6f}, Dual {dual_res:.6f}")
        print(f"Gap relativo: {gap_rel2:.6f}\n")

    # Resultado final
    print("Solução final (Preditor-Corretor):")
    print(f"Solução primal (x): {x}, Valor primal: {primal_value:.4f}")
    print(f"Solução dual (y): {y}, Valor dual: {dual_value:.4f}\n")
    return x, y_values, z, values

def plot_convergencia(values_affine, values_affine_dual, values_classic, values_classic_dual, values_pc, values_pc_dual):
    # Afim-Escala
    plt.figure(figsize=(10, 6))
    plt.plot(values_affine, label='Valor Primal', marker='o')
    plt.plot(values_affine_dual, label='Valor Dual', marker='o')
    plt.xlabel('Iterações')
    plt.ylabel('Valor da Função Objetivo')
    plt.title('Primal-Dual Afim-Escala')
    plt.legend()
    plt.grid()
    plt.show()

    # Clássico
    plt.figure(figsize=(10, 6))
    plt.plot(values_classic, label='Valor Primal', marker='o')
    plt.plot(values_classic_dual, label='Valor Dual', marker='o')
    plt.xlabel('Iterações')
    plt.ylabel('Valor da Função Objetivo')
    plt.title('Primal-Dual Clássico')
    plt.legend()
    plt.grid()
    plt.show()

    # Preditor-Corretor
    plt.figure(figsize=(10, 6))
    plt.plot(values_pc, label='Valor Primal', marker='o')
    plt.plot(values_pc_dual, label='Valor Dual', marker='o')
    plt.xlabel('Iterações')
    plt.ylabel('Valor da Função Objetivo')
    plt.title('Primal-Dual Preditor-Corretor')
    plt.legend()
    plt.grid()
    plt.show()


# Definição do Problema
m, n = 10, 10

A = np.array([
    [10, 2, 3, 1, 1, 2, 3, 1, 2, 3],
    [2, 15, 2, 3, 2, 3, 1, 2, 3, 1],
    [3, 2, 20, 1, 3, 1, 2, 3, 1, 2],
    [1, 3, 1, 25, 2, 3, 1, 2, 3, 2],
    [1, 2, 3, 2, 30, 1, 3, 1, 2, 3],
    [2, 3, 1, 3, 1, 18, 2, 3, 1, 2],
    [3, 1, 2, 1, 3, 2, 22, 1, 3, 1],
    [1, 2, 3, 2, 1, 3, 1, 26, 2, 3],
    [2, 3, 1, 3, 2, 1, 3, 2, 28, 1],
    [3, 1, 2, 2, 3, 2, 1, 3, 1, 35]
])

b = np.array([50, 85, 75, 95, 100, 90, 80, 70, 110, 120])
c = np.array([7, 3, 9, 2, 5, 1, 8, 4, 6, 10])

x0 = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0])
y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
z0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

values_affine, values_affine_dual = [], []
values_classic, values_classic_dual = [], []
values_pc, values_pc_dual = [], []

x_affine, y_affine, z_affine, values_affine = primal_dual_afim_escala(A, b, c, x0, y0, z0)
values_affine_dual = [b.T @ y for y in y_affine]

x_classic, y_classic, z_classic, values_classic = primal_dual_classico(A, b, c, x0, y0, z0)
values_classic_dual = [b.T @ y for y in y_classic]

x_pc, y_pc, z_pc, values_pc = primal_dual_preditor_corrector(A, b, c, x0, y0, z0)
values_pc_dual = [b.T @ y for y in y_pc]

plot_convergencia(values_affine, values_affine_dual, values_classic, values_classic_dual, values_pc, values_pc_dual)