"""Reimplementacion independiente del estimador CEFI (revision R3).
No importa nada del repositorio auditado salvo los datos.
Objetivo (identico al del manuscrito y al de cuda_stiefel.evaluate_batch_cefi):
   f(W) = 0.5*[ logdet(s2 * Am Am^T + Sm) - logdet(Sm) ],  Am = W A W^T, Sm = W S W^T
   CEFI = max_q [ f_q^* / q - EI_p / p ]
"""
import numpy as np


def ledoit_wolf(residuals):
    T, p = residuals.shape
    S = (residuals.T @ residuals) / T
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    d2 = np.sum((S - target) ** 2)
    X2 = residuals ** 2
    b_bar_sq = float((np.sum(X2.T @ X2) - T * np.sum(S ** 2)) / (T ** 2))
    b2 = min(b_bar_sq, d2)
    delta = 0.0 if d2 <= 0 else max(0.0, min(1.0, b2 / d2))
    out = (1.0 - delta) * S + delta * target
    return 0.5 * (out + out.T)


def fit_var1(X, lam0=1e-4):
    X = X - X.mean(axis=0, keepdims=True)
    p = X.shape[1]
    XL, XN = X[:-1], X[1:]
    XtX = XL.T @ XL
    XtY = XL.T @ XN
    scale = np.trace(XtX) / p
    AT = np.linalg.solve(XtX + lam0 * scale * np.eye(p), XtY)
    resid = XN - XL @ AT
    return AT.T, ledoit_wolf(resid)


def _clean(S, scale):
    p = S.shape[-1]
    return 0.5 * (S + S.T) + 1e-10 * scale * np.eye(p)


def ei_micro(A, S, s2):
    p = A.shape[0]
    M = s2 * (A @ A.T) + S
    return 0.5 * (np.linalg.slogdet(M)[1] - np.linalg.slogdet(S)[1])


def obj(W, A, S, s2):
    Am = W @ A @ W.T
    Sm = W @ S @ W.T
    M = s2 * (Am @ Am.T) + Sm
    return 0.5 * (np.linalg.slogdet(M)[1] - np.linalg.slogdet(Sm)[1])


def grad_obj(W, A, S, s2):
    """Gradiente euclideo analitico de obj respecto de W (q x p)."""
    Am = W @ A @ W.T
    Sm = W @ S @ W.T
    M = s2 * (Am @ Am.T) + Sm
    G = np.linalg.inv(M)
    Sinv = np.linalg.inv(Sm)
    P = s2 * (G @ Am)          # df/dAm
    Q = 0.5 * (G - Sinv)       # df/dSm
    return (P @ W @ A.T + P.T @ W @ A) + (Q @ W @ S.T + Q.T @ W @ S)


def retract(W):
    Qm, R = np.linalg.qr(W.T)
    signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
    return (Qm * signs).T


def optimize_q(A, S, s2, q, n_restarts=12, max_iter=100, lr=0.05,
               Sigma_x=None, adaptive=False, seed_base=42):
    p = A.shape[0]
    inits = []
    if Sigma_x is not None:
        _, ev = np.linalg.eigh(0.5 * (Sigma_x + Sigma_x.T))
        inits.append(ev[:, -q:].T.copy())
    for r in range(1, n_restarts):
        rng = np.random.default_rng(seed_base + r * 1000 + q * 17)
        Qm, R = np.linalg.qr(rng.standard_normal((p, q)))
        signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
        inits.append((Qm * signs).T.copy())
    best = -np.inf
    bestW = None
    for W in inits:
        step = lr
        fcur = obj(W, A, S, s2)
        for _ in range(max_iter):
            g = grad_obj(W, A, S, s2)
            gr = g - W @ g.T @ W          # proyeccion tangente (misma que el repo)
            if not adaptive:
                W = retract(W + step * gr)
                continue
            ok = False
            for _ls in range(30):
                Wn = retract(W + step * gr)
                fn = obj(Wn, A, S, s2)
                if fn > fcur:
                    W, fcur, ok = Wn, fn, True
                    step *= 1.5
                    break
                step *= 0.5
            if not ok:
                break
        f = obj(W, A, S, s2)
        if f > best:
            best, bestW = f, W
    return best, bestW


def cefi(A, S, Sigma_x, kappa=1.0, qs=None, tie_tol=1e-7, **kw):
    p = A.shape[0]
    scale = max(np.trace(Sigma_x) / p, 1e-12)
    s2 = (kappa ** 2) * scale
    Sc = _clean(S, scale)
    eim = ei_micro(A, Sc, s2)
    dens = eim / p
    if qs is None:
        qs = list(range(1, p))
    spec = {}
    for q in qs:
        f, _ = optimize_q(A, Sc, s2, q, Sigma_x=Sigma_x, **kw)
        spec[q] = f / q - dens
    mx = max(spec.values())
    qstar = min(q for q, v in spec.items() if v >= mx - tie_tol)
    return spec[qstar], qstar, spec, eim


def cefi_svd_bound(A, S, Sigma_x, kappa=1.0):
    """Cota superior analitica (Yang et al.): modos singulares del operador blanqueado."""
    p = A.shape[0]
    scale = max(np.trace(Sigma_x) / p, 1e-12)
    s2 = (kappa ** 2) * scale
    Sc = _clean(S, scale)
    L = np.linalg.cholesky(Sc)
    K = np.linalg.solve(L, A)
    sv = np.linalg.svd(K, compute_uv=False)
    terms = np.log1p(s2 * sv ** 2)
    eim = 0.5 * terms.sum()
    dens = eim / p
    spec = {q: 0.5 * terms[:q].sum() / q - dens for q in range(1, p)}
    qs = max(spec, key=spec.get)
    return spec[qs], qs, spec
