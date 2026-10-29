import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelo import Classificador, dados_para_matrizes, segmentos_decisao
from dados import exemplo, salvar, carregar, validar_ponto


class ClassificacaoTest(unittest.TestCase):
    def setUp(self):
        self.points=exemplo('tres')
        self.X,self.y,self.classes=dados_para_matrizes(self.points)
        self.net=Classificador(self.classes)

    def test_initial_predictions_and_loss(self):
        p=self.net.probabilidades(self.X)
        np.testing.assert_allclose(p.sum(axis=1),1)
        np.testing.assert_allclose(p,1/3)
        self.assertAlmostEqual(self.net.avaliar(self.X,self.y)[0],math.log(3))

    def test_gradients_match_finite_differences(self):
        self.net.W[:]=[[.3,-.1,.2],[-.2,.4,-.15]]
        self.net.b[:]=[.1,-.2,.3]
        dw,db=self.net.gradientes(self.X,self.y)
        for array,index,gradient in [(self.net.W,(0,1),dw),(self.net.W,(1,2),dw),(self.net.b,(0,),db)]:
            old=array[index];eps=1e-6
            array[index]=old+eps;plus=self.net.avaliar(self.X,self.y)[0]
            array[index]=old-eps;minus=self.net.avaliar(self.X,self.y)[0]
            array[index]=old
            self.assertAlmostEqual((plus-minus)/(2*eps),gradient[index],places=7)

    def test_demo_learns_and_probabilities_change(self):
        initial_loss,initial_acc,_=self.net.avaliar(self.X,self.y)
        initial_weights=self.net.W.copy()
        _,first_acc,_=self.net.passo(self.X,self.y,.5)
        for _ in range(199):loss,acc,hits=self.net.passo(self.X,self.y,.5)
        self.assertLess(loss,initial_loss*.4)
        self.assertGreater(acc,first_acc)
        self.assertGreaterEqual(acc,.98)
        self.assertFalse(np.array_equal(self.net.W,initial_weights))
        self.assertEqual(hits,len(self.X))
        self.assertTrue(np.isfinite(self.net.probabilidades([0,0])).all())

    def test_true_decision_boundaries_are_clipped_to_winning_classes(self):
        self.net.W=np.array([[10.,-10.,0.],[0.,0.,10.]])
        self.net.b[:]=0
        segments=segmentos_decisao(self.net)
        ab=[s for s in segments if s[:2]==('A','B')]
        self.assertEqual(len(ab),1)
        endpoints=[np.array(ab[0][2]),np.array(ab[0][3])]
        for p in endpoints:self.assertAlmostEqual(p[0],0)
        self.assertAlmostEqual(min(p[1] for p in endpoints),-10)
        self.assertAlmostEqual(max(p[1] for p in endpoints),0)
        for a,b,p0,p1 in segments:
            mid=(np.array(p0)+np.array(p1))/2
            probs=self.net.probabilidades(mid)[0]
            i=self.classes.index(a);j=self.classes.index(b)
            self.assertAlmostEqual(probs[i],probs[j],places=10)
            self.assertGreaterEqual(probs[i]+1e-10,float(np.max(probs)))

    def test_xor_does_not_become_perfect_with_linear_classifier(self):
        X,y,c=dados_para_matrizes(exemplo('xor'))
        net=Classificador(c)
        for _ in range(500):loss,acc,hits=net.passo(X,y,.5)
        self.assertLess(acc,1.0)
        self.assertLess(hits,len(X))

    def test_project_roundtrip_and_continuation(self):
        hist=[]
        for i in range(12):
            loss,acc,hits=self.net.passo(self.X,self.y,.5)
            hist.append({'passo':i+1,'perda':loss,'acuracia':acc,'acertos':hits})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'projeto.json'
            salvar(path,self.points,self.net,hist,{'taxa':.5})
            ps,net,history,cfg=carregar(path)
            self.assertEqual(ps,self.points);self.assertEqual(history,hist)
            np.testing.assert_allclose(self.net.probabilidades(self.X),net.probabilidades(self.X),atol=0,rtol=0)
            np.testing.assert_allclose(self.net.passo(self.X,self.y,.5),net.passo(self.X,self.y,.5),atol=0,rtol=0)

    def test_rejects_invalid_inputs(self):
        for p in [{'x1':float('nan'),'x2':0,'classe':'A'},
                  {'x1':11,'x2':0,'classe':'A'},
                  {'x1':0,'x2':0,'classe':'D'}]:
            with self.assertRaises(ValueError):validar_ponto(p)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json';path.write_text(json.dumps({'formato':'errado'}))
            with self.assertRaises(ValueError):carregar(path)


if __name__=='__main__':unittest.main()
