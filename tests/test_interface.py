import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pygame
from classificacao import App


class InterfaceTest(unittest.TestCase):
    def setUp(self):
        self.app=App((1600,900));self.app.render()
    def tearDown(self):pygame.quit()

    def click(self,name):
        self.app.render();rect,_,enabled=self.app.buttons[name];self.assertTrue(enabled,name)
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,{'button':1,'pos':rect.center}))

    def click_graph(self,x,y,button=1):
        self.app.render();pos=self.app.world_to_screen(x,y)
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,{'button':button,'pos':pos}))

    def test_user_flow_train_infer_and_add(self):
        self.assertEqual(len(self.app.points),54)
        self.click('step');self.assertEqual(len(self.app.history),1)
        first=self.app.history[0]['acuracia']
        self.app.advance(199);self.assertGreater(self.app.history[-1]['acuracia'],first)
        self.click('predict_mode');before=len(self.app.points)
        self.click_graph(-1,0)
        self.assertEqual(len(self.app.points),before)
        self.assertIsNotNone(self.app.test_point)
        probs=self.app.net.probabilidades(self.app.test_point)[0]
        self.assertEqual(self.app.net.classes[int(probs.argmax())],'C')
        self.click('add_mode');self.click_graph(0,8)
        self.assertEqual(len(self.app.points),before+1)
        self.assertIsNone(self.app.net)

    def test_click_remove_and_reset(self):
        self.click('step')
        p=self.app.points[0]
        self.click_graph(p['x1'],p['x2'],button=3)
        self.assertEqual(len(self.app.points),53)
        self.assertIsNone(self.app.net)

    def test_modal_switch_examples_and_no_single_class_training(self):
        self.click('two');self.assertIsNotNone(self.app.modal)
        self.click('modal_yes')
        self.assertEqual({p['classe'] for p in self.app.points},{'A','B'})
        self.app.points=[p for p in self.app.points if p['classe']=='A']
        self.click('step');self.assertIsNone(self.app.net)
        self.assertIn('duas classes',self.app.status)

    def test_numeric_input_and_invalid_coordinates(self):
        self.click('input_x1')
        for c in '-2.5':
            self.app.handle_event(pygame.event.Event(pygame.KEYDOWN,{'key':ord(c),'unicode':c}))
        self.app.coord['x2']='5,5'
        self.click('coord_action')
        p=self.app.points[-1]
        self.assertEqual((p['x1'],p['x2']),(-2.5,5.5))
        count=len(self.app.points);self.app.coord['x1']='900'
        self.click('coord_action')
        self.assertEqual(len(self.app.points),count)

    def test_project_save_open_and_continue(self):
        self.app.step();self.app.advance(20);self.app.set_mode('prever');self.app.set_test(1,2)
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'p.json')
            with patch.object(self.app,'file_dialog',return_value=path):self.app.save_file()
            saved=self.app.net.para_dict()
            self.app.clear_points()
            with patch.object(self.app,'file_dialog',return_value=path):self.app.open_file()
            self.assertEqual(self.app.net.para_dict(),saved)
            self.assertEqual(self.app.test_point,(1,2))
            self.assertEqual(len(self.app.history),21)
            self.app.step();self.assertEqual(len(self.app.history),22)

    def test_invalid_open_preserves_project(self):
        old=list(self.app.points)
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/'invalid.json';bad.write_text('{"bad":true}')
            with patch.object(self.app,'file_dialog',return_value=str(bad)):self.app.open_file()
        self.assertEqual(self.app.points,old)
        self.assertIn('Não foi possível',self.app.status)

    def test_scaled_window_hit_detection_and_event_loop(self):
        pygame.display.set_mode((1280,720),pygame.RESIZABLE)
        self.app.render()
        rect=self.app.buttons['step'][0]; x0,y0,s=self.app.viewport
        pos=(int(x0+s*rect.centerx),int(y0+s*rect.centery))
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,{'button':1,'pos':pos}))
        self.assertEqual(len(self.app.history),1)
        self.app.start()
        def stop():
            time.sleep(.35);pygame.event.post(pygame.event.Event(pygame.QUIT))
        threading.Thread(target=stop,daemon=True).start()
        self.app.run()
        self.assertGreaterEqual(len(self.app.history),3)


if __name__=='__main__':unittest.main()
