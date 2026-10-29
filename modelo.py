"""Classificador linear multiclasse (softmax), treinado por gradiente em lote."""
from __future__ import annotations
import numpy as np

ESCALA = 10.0
ROTULOS = ('A', 'B', 'C')


def dados_para_matrizes(pontos, classes=None):
    classes = list(classes) if classes is not None else sorted({p['classe'] for p in pontos})
    ind = {c: i for i, c in enumerate(classes)}
    x = np.asarray([[p['x1'], p['x2']] for p in pontos], dtype=np.float64)
    y = np.asarray([ind[p['classe']] for p in pontos], dtype=np.int64)
    return x, y, classes


class Classificador:
    """Três ou duas classes; pesos usam x1/10 e x2/10 como entradas."""
    def __init__(self, classes):
        if not 2 <= len(classes) <= 3 or len(set(classes)) != len(classes) or any(c not in ROTULOS for c in classes):
            raise ValueError('Use duas ou três classes distintas entre A, B e C.')
        self.classes = list(classes)
        self.W = np.zeros((2, len(classes)), dtype=np.float64)
        self.b = np.zeros(len(classes), dtype=np.float64)

    def probabilidades(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
            raise ValueError('Cada ponto deve conter dois números finitos.')
        z = (X / ESCALA) @ self.W + self.b
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def avaliar(self, X, y):
        y = np.asarray(y, dtype=int)
        if len(y) == 0:
            raise ValueError('Não há pontos para avaliar.')
        probs = self.probabilidades(X)
        if len(y) != len(probs) or np.any(y < 0) or np.any(y >= len(self.classes)):
            raise ValueError('Rótulos incompatíveis com os pontos.')
        loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean()
        correct = int(np.sum(np.argmax(probs, axis=1) == y))
        return float(loss), correct / len(y), correct

    def gradientes(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        probs = self.probabilidades(X)
        if len(y) == 0 or len(y) != len(probs) or np.any(y < 0) or np.any(y >= len(self.classes)):
            raise ValueError('Dados e rótulos incompatíveis.')
        dif = probs.copy()
        dif[np.arange(len(y)), y] -= 1
        dif /= len(y)
        return (X / ESCALA).T @ dif, dif.sum(axis=0)

    def passo(self, X, y, taxa):
        if not np.isfinite(taxa) or not 0 < taxa <= 2:
            raise ValueError('Taxa de aprendizado inválida.')
        dw, db = self.gradientes(X, y)
        self.W -= taxa * dw
        self.b -= taxa * db
        return self.avaliar(X, y)

    def para_dict(self):
        return {'classes': self.classes, 'W': self.W.tolist(), 'b': self.b.tolist()}

    @classmethod
    def de_dict(cls, obj):
        m = cls(obj['classes'])
        W, b = np.asarray(obj['W'], dtype=float), np.asarray(obj['b'], dtype=float)
        if W.shape != (2, len(m.classes)) or b.shape != (len(m.classes),) or not (np.isfinite(W).all() and np.isfinite(b).all()):
            raise ValueError('Pesos ou vieses inválidos.')
        m.W, m.b = W, b
        return m


def segmentos_decisao(modelo, limite=10.0):
    """Segmentos exatos onde duas classes empatam e não perdem para outra.

    As desigualdades são recortadas contra o quadrado [-limite, limite]^2.
    Cada retorno é (classe_i, classe_j, (x1,y1), (x2,y2)).
    """
    result=[]
    for i in range(len(modelo.classes)):
        for j in range(i+1,len(modelo.classes)):
            a,b=(modelo.W[:,i]-modelo.W[:,j])/ESCALA
            c=float(modelo.b[i]-modelo.b[j])
            if abs(a)<1e-12 and abs(b)<1e-12:
                continue
            points=[]
            if abs(b)>1e-12:
                for x in (-limite,limite):
                    y=-(a*x+c)/b
                    if -limite-1e-9<=y<=limite+1e-9:points.append((float(x),float(y)))
            if abs(a)>1e-12:
                for y in (-limite,limite):
                    x=-(b*y+c)/a
                    if -limite-1e-9<=x<=limite+1e-9:points.append((float(x),float(y)))
            unique=[]
            for p in points:
                if not any(np.linalg.norm(np.asarray(p)-q)<1e-8 for q in unique):
                    unique.append(np.asarray(p))
            if len(unique)<2:continue
            p0,p1=max(((u,v) for idx,u in enumerate(unique) for v in unique[idx+1:]),key=lambda q:np.linalg.norm(q[0]-q[1]))
            lower,upper=0.0,1.0
            for k in range(len(modelo.classes)):
                if k in (i,j):continue
                delta=(modelo.W[:,i]-modelo.W[:,k])/ESCALA
                bias=float(modelo.b[i]-modelo.b[k])
                f0=float(p0@delta+bias)
                slope=float((p1-p0)@delta)
                if abs(slope)<1e-12:
                    if f0 < -1e-10:lower,upper=1,0
                    continue
                bound=-f0/slope
                if slope>0:lower=max(lower,bound)
                else:upper=min(upper,bound)
            if lower<=upper+1e-10 and upper>=0 and lower<=1:
                q0=p0+max(0,lower)*(p1-p0)
                q1=p0+min(1,upper)*(p1-p0)
                result.append((modelo.classes[i],modelo.classes[j],tuple(q0),tuple(q1)))
    return result
