import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Função objetivo - Inserida manualmente, bem como as derivadas utilizadas nos Gradientes e Hessianas.
def func_obj(x):
    return x[0]**3 + x[1]**3 + x[0] * x[1]

# Restrições de desigualdade
def restr_inequacao(x):
    return np.array([x[0] + x[1] - 1])  # g1(x) <= 0

# Restrições de igualdade
def restr_igualdade(x):
    return np.array([x[0]**2 + x[1]**2 - 1])  # h1(x) = 0

# Gradientes
def grad_obj(x):
    return np.array([3 * x[0]**2 + x[1],    # df/dx1
                     3 * x[1]**2 + x[0]])   # df/dx2

def grad_inequacao(x):
    return np.array([[1, 1]])  # Gradiente de g1(x)

def grad_igualdade(x):
    return np.array([[2 * x[0], 2 * x[1]]])   # Gradiente de h1(x)

# Hessianas
def hess_obj(x):
    return np.array([[6 * x[0], 1], [1, 6 * x[1]]])  # Segunda derivada de f(x) em relação a x1 e x2

def hess_igualdade(x):
    return np.array([[2, 0], [0, 2]])  # Segunda derivada de g1 em relação a x1 e x2

def hess_inequacao(x):
    return np.zeros((2, 2))  # h1(x) é linear, Hessiana é zero


# Barreira Logarítmica
def barreira_logaritmica(func_obj, restr_inequacao, grad_obj, grad_inequacao, hess_obj, x0, mu0, beta, epsilon):
    x = x0.copy()
    mu = mu0
    valores_f = []
    f_prev = np.inf

    while mu > epsilon:
        def phi(x):
            g = restr_inequacao(x)
            if np.any(g >= 0):
                return np.inf  # Penaliza violações das restrições
            return func_obj(x) - mu * np.sum(np.log(-g))

        def grad_phi(x):
            g = restr_inequacao(x)
            grad_g = grad_inequacao(x)
            grad_pen = -mu * np.sum(grad_g / g[:, None], axis=0)
            return grad_obj(x) + grad_pen

        def hess_phi(x):
            g = restr_inequacao(x)
            grad_g = grad_inequacao(x)
            hess_pen = mu * np.sum((grad_g.T @ grad_g) / (g[:, None] * g[None, :]), axis=0)
            return hess_obj(x) + hess_pen

        # Método de Newton
        for _ in range(50):
            grad = grad_phi(x)
            hess = hess_phi(x)
            delta_x = solve(hess, -grad)
            x = x + delta_x
            f_curr = func_obj(x)
            
            # Critério de parada - considera a variação na função objetivo
            if np.linalg.norm(grad) < epsilon or abs(f_curr - f_prev) < epsilon:
                break

            f_prev = f_curr

        valores_f.append(func_obj(x))
        print(f"x: {x}, f(x): {func_obj(x)}")
        mu *= beta  # Reduz o parâmetro de barreira

    print("Solução final:", x)
    print("Valor da função objetivo na solução final:", func_obj(x))

    return valores_f

# Preditor-Corretor
def barreira_logaritmica_preditor_corretor(func_obj, restr_inequacao, restr_igualdade, grad_obj, grad_inequacao, grad_igualdade, hess_obj, hess_inequacao, hess_igualdade, x0, s0, pi0, lambda0, mu0, beta, epsilon, regularizacao=1e-8):
    x = x0.copy()
    s = np.maximum(s0.copy(), 1e-3)
    pi = pi0.copy()
    lamb = np.maximum(lambda0.copy(), 1e-3)
    mu = mu0
    valores_f = []
    f_prev = np.inf

    while mu > epsilon:
        def residuos(x, s, lamb, pi):
            g = restr_inequacao(x)
            h = restr_igualdade(x)
            grad_f = grad_obj(x)
            grad_g = grad_inequacao(x)
            grad_h = grad_igualdade(x)

            r_dual = grad_f + grad_g.T @ lamb + grad_h.T @ pi
            r_cent = -np.diag(lamb) @ s - mu * np.ones_like(s)
            r_pri_ineq = g + s
            r_pri_eq = h

            return r_dual, r_cent, r_pri_ineq, r_pri_eq

        def sistema_newton(x, s, lamb, pi, r_dual, r_cent, r_pri_ineq, r_pri_eq):
            grad_g = grad_inequacao(x)
            grad_h = grad_igualdade(x)
            hess_f = hess_obj(x)

            n = len(x)
            m = len(s)

            # Sistema linear expandido
            KKT = np.block([
                [hess_f, grad_g.T, grad_h.T, np.zeros((n, m))],
                [-np.diag(lamb) @ grad_g, -np.diag(s), np.zeros((m, len(pi))), -np.eye(m)],
                [grad_g, np.zeros((m, m)), np.zeros((m, len(pi))), np.zeros((m, m))],
                [grad_h, np.zeros((len(pi), m)), np.eye(len(pi)), np.zeros((len(pi), m))]
            ])

            KKT += np.eye(KKT.shape[0]) * regularizacao

            rhs = -np.concatenate([r_dual, r_cent, r_pri_ineq, r_pri_eq])

            delta = solve(KKT, rhs)

            delta_x = delta[:n]
            delta_s = delta[n:n + m]
            delta_lamb = delta[n + m:n + 2 * m]
            delta_pi = delta[n + 2 * m:]

            return delta_x, delta_s, delta_lamb, delta_pi

        # Resíduos
        r_dual, r_cent, r_pri_ineq, r_pri_eq = residuos(x, s, lamb, pi)

        # Passo preditor
        delta_x, delta_s, delta_lamb, delta_pi = sistema_newton(x, s, lamb, pi, r_dual, r_cent, r_pri_ineq, r_pri_eq)

        # Atualização das variáveis
        x += delta_x
        s = np.maximum(s + delta_s, 1e-6)  # Garantir s > 0
        lamb = np.maximum(lamb + delta_lamb, 1e-6)  # Garantir lambda > 0
        pi += delta_pi
        f_curr = func_obj(x)

       # Critério de parada - considera a variação na função objetivo
        if np.linalg.norm(r_dual) < epsilon or abs(f_curr - f_prev) < epsilon:
            break                 # desativar critério de parada para análise de convergência                                               

        f_prev = f_curr

        valores_f.append(func_obj(x))
        print(f"x: {x}, f(x): {func_obj(x)}")

        # Atualizar parâmetro de barreira
        mu *= beta

    print("Solução final:", x)
    print("Valor da função objetivo na solução final:", func_obj(x))

    return valores_f


# Parâmetros compartilhados
x0 = np.array([1.0, 1.0])
s0 = np.array([1.0])
pi0 = np.array([1.0])
lambda0 = np.array([1.0])
mu0 = 1.0
beta = 0.5
epsilon = 1e-3

# Barreira Logarítmica
valores_BL = barreira_logaritmica(
    func_obj,
    restr_inequacao,
    grad_obj,
    grad_inequacao,
    hess_obj,
    x0,
    mu0,
    beta,
    epsilon
)

# Barreira Logarítmica Preditor-Corretor
valores_BLPC = barreira_logaritmica_preditor_corretor(
    func_obj,
    restr_inequacao,
    restr_igualdade,
    grad_obj,
    grad_inequacao,
    grad_igualdade,
    hess_obj,
    hess_inequacao,
    hess_igualdade,
    x0,
    s0,
    pi0,
    lambda0,
    mu0,
    beta,
    epsilon
)

# Análise de Convergência
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(valores_BL) + 1), valores_BL, marker='o', label='Barreira Logarítmica')
plt.plot(range(1, len(valores_BLPC) + 1), valores_BLPC, marker='o', label='Barreira Logarítmica Preditor-Corretor')
plt.xlabel('Iterações', fontsize = 16)
plt.ylabel('Função Objetivo', fontsize = 16)
plt.title('Análise de Convergência', fontsize = 18)
plt.legend(fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()