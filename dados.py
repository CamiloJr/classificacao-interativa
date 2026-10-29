"""Pontos sintéticos e leitura/gravação segura dos projetos JSON."""
from __future__ import annotations
import json
import math
import os
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from modelo import Classificador, ROTULOS

LIMITE = 300


def validar_ponto(p):
    if not isinstance(p, dict) or p.get('classe') not in ROTULOS:
        raise ValueError('Classe de ponto inválida.')
    x1, x2 = float(p['x1']), float(p['x2'])
    if not all(math.isfinite(z) and -10 <= z <= 10 for z in (x1, x2)):
        raise ValueError('As coordenadas precisam estar entre -10 e 10.')
    return {'x1': x1, 'x2': x2, 'classe': p['classe']}


def exemplo(tipo):
    rng = np.random.default_rng(2026)
    if tipo == 'binario':
        centers = [('A', -4.5, -3.5), ('B', 4.2, 3.0)]
    elif tipo == 'tres':
        centers = [('A', -5, -4), ('B', 4, -2), ('C', -1, 0)]
    elif tipo == 'xor':
        centers = [('A', -5, -5), ('A', 5, 5), ('B', -5, 5), ('B', 5, -5)]
    else:
        raise ValueError('Exemplo desconhecido.')
    n = 14 if tipo == 'xor' else 18
    spread = 1.05 if tipo == 'xor' else .9 if tipo == 'tres' else 1.4
    result = []
    for label, a, b in centers:
        values = rng.normal((a, b), spread, size=(n, 2))
        values = np.clip(values, -9.6, 9.6)
        result.extend({'x1': round(float(x), 2), 'x2': round(float(y), 2), 'classe': label} for x, y in values)
    return result


def contagens(pontos):
    count = Counter(p['classe'] for p in pontos)
    return {c: count[c] for c in ROTULOS}


def salvar(path, pontos, modelo, historico, config):
    obj = {'formato': 'classificacao-softmax-2d', 'versao': 1,
           'pontos': pontos, 'modelo': modelo.para_dict() if modelo else None,
           'historico': historico, 'configuracao': config}
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.classificacao_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def carregar(path):
    if Path(path).stat().st_size > 10 * 1024 * 1024:
        raise ValueError('Arquivo maior que 10 MB.')
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or obj.get('formato') != 'classificacao-softmax-2d' or obj.get('versao') != 1:
        raise ValueError('Arquivo de projeto incompatível.')
    raw = obj.get('pontos')
    if not isinstance(raw, list) or len(raw) > LIMITE:
        raise ValueError('Lista de pontos inválida.')
    pontos = [validar_ponto(p) for p in raw]
    model = Classificador.de_dict(obj['modelo']) if obj.get('modelo') is not None else None
    if model and set(model.classes) != {p['classe'] for p in pontos}:
        raise ValueError('Classes do modelo não correspondem aos pontos.')
    raw_history = obj.get('historico', [])
    if not isinstance(raw_history, list) or len(raw_history) > 5000 or (raw_history and not model):
        raise ValueError('Histórico inválido.')
    history = []
    for i, e in enumerate(raw_history, 1):
        if not isinstance(e, dict) or e.get('passo') != i:
            raise ValueError('Sequência de passos inválida.')
        loss, acc = float(e['perda']), float(e['acuracia'])
        correct = int(e['acertos'])
        if not (math.isfinite(loss) and loss >= 0 and math.isfinite(acc) and 0 <= acc <= 1 and 0 <= correct <= len(pontos)):
            raise ValueError('Métricas inválidas.')
        history.append({'passo': i, 'perda': loss, 'acuracia': acc, 'acertos': correct})
    cfg = obj.get('configuracao', {})
    if not isinstance(cfg, dict):
        raise ValueError('Configuração inválida.')
    return pontos, model, history, cfg
