"""Classificação supervisionada 2D com treinamento e visualização interativos."""
from __future__ import annotations
import ctypes
import math
import sys
import time
from pathlib import Path
import numpy as np

try:
    import pygame
except ImportError as exc:
    print('Dependência ausente. Execute: python -m pip install -r requirements.txt')
    raise SystemExit(1) from exc
from modelo import Classificador, dados_para_matrizes, segmentos_decisao, ROTULOS
from dados import LIMITE, validar_ponto, exemplo, contagens, salvar, carregar

W, H, FPS = 1600, 900, 60
BG=(10,12,17); PANEL=(19,23,31); DARK=(13,16,22); BORDER=(54,63,76)
TEXT=(235,240,246); MUTED=(155,168,183); DIM=(102,118,133)
CYAN=(70,211,220); GREEN=(88,208,145); RED=(244,116,116); GOLD=(246,202,91)
CLASS_COLORS={'A':(67,205,224),'B':(246,159,70),'C':(194,149,246)}
BUTTON=(42,50,62); HOVER=(60,72,86)


def dpi_awareness():
    if sys.platform == 'win32':
        try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try: ctypes.windll.user32.SetProcessDPIAware()
            except Exception: pass


def blend(a,b,t):
    return tuple(int((1-t)*x+t*y) for x,y in zip(a,b))


class App:
    def __init__(self, size=None):
        dpi_awareness()
        pygame.init()
        pygame.display.set_caption('Classificação Interativa | Aprendizado Supervisionado')
        info = pygame.display.Info()
        if size is None:
            dw = info.current_w if info.current_w >= 800 else W
            dh = info.current_h if info.current_h >= 600 else H
            s = min(1, dw*.96/W, dh*.92/H)
            size = (max(800,int(W*s)), max(450,int(H*s)))
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.canvas = pygame.Surface((W,H))
        self.clock = pygame.time.Clock()
        self.fonts = {
            'title':pygame.font.SysFont('segoeui,arial',29,bold=True),
            'head':pygame.font.SysFont('segoeui,arial',20,bold=True),
            'bold':pygame.font.SysFont('segoeui,arial',17,bold=True),
            'body':pygame.font.SysFont('segoeui,arial',17),
            'small':pygame.font.SysFont('segoeui,arial',14),
            'tiny':pygame.font.SysFont('segoeui,arial',12),
            'mono':pygame.font.SysFont('consolas,monospace',15),
            'big':pygame.font.SysFont('segoeui,arial',32,bold=True),
        }
        self.plot = pygame.Rect(91, 247, 733, 453)
        self.points = exemplo('tres')
        self.selected_class = 'A'
        self.mode = 'adicionar'
        self.test_point = None
        self.coord = {'x1':'0', 'x2':'0'}
        self.focus = None
        self.select_all = False
        self.net = None
        self.history=[]
        self.X=self.y=None
        self.rates=[.01,.03,.1,.3,.5,1.0,2.0]
        self.rate=.5
        self.max_steps_options=[100,300,500,1000,2000,5000]
        self.max_steps=500
        self.time_options=[.01,.05,.1,.25,.5,1.0,2.0,5.0]
        self.interval=.1
        self.running=False
        self.fast=False
        self.last_tick=0.0
        self.show_regions=True
        self.model_version=0
        self.region_cache=None
        self.cache_key=None
        self.status='Exemplo com três classes carregado. Clique em Passo ou Treinar.'
        self.viewport=(0,0,1)
        self.buttons={}
        self.modal=None
        self.alive=True

    def text(self, value, x,y, style='body', color=TEXT, max_width=None):
        value=str(value); f=self.fonts[style]
        if max_width is not None and f.size(value)[0]>max_width:
            while value and f.size(value+'…')[0] > max_width: value=value[:-1]
            value+='…'
        self.canvas.blit(f.render(value,True,color),(int(x),int(y)))

    def center(self, value, rect, style='body', color=TEXT):
        im=self.fonts[style].render(str(value),True,color)
        self.canvas.blit(im, im.get_rect(center=pygame.Rect(rect).center))

    def panel(self,rect,color=PANEL,border=BORDER,radius=11):
        rect=pygame.Rect(rect)
        pygame.draw.rect(self.canvas,color,rect,border_radius=radius)
        pygame.draw.rect(self.canvas,border,rect,1,border_radius=radius)

    def logical(self,pos):
        x,y,s=self.viewport
        return int((pos[0]-x)/s), int((pos[1]-y)/s)

    def button(self,key,label,rect,action,kind='normal',enabled=True,style='small'):
        rect=pygame.Rect(rect)
        hover=rect.collidepoint(self.logical(pygame.mouse.get_pos())) and enabled
        if not enabled: bg,fg=(31,36,44),(96,108,121)
        elif kind=='primary': bg,fg=((51,172,183) if hover else (34,140,153)),(255,255,255)
        elif kind=='active': bg,fg=(42,99,123),TEXT
        elif kind=='danger': bg,fg=((101,56,61) if hover else (71,44,49)),(250,208,208)
        else: bg,fg=(HOVER if hover else BUTTON),TEXT
        pygame.draw.rect(self.canvas,bg,rect,border_radius=6)
        pygame.draw.rect(self.canvas,BORDER,rect,1,border_radius=6)
        self.center(label,rect,style,fg)
        self.buttons[key]=(rect,action,enabled)

    def header(self):
        self.text('Classificação Interativa',26,20,'title')
        self.text('Aprendizado supervisionado • pontos, classes e fronteiras de decisão',28,59,'body',MUTED)
        self.text('AULA 6  •  REGRESSÃO LOGÍSTICA MULTICLASSE',934,27,'small',CYAN)
        self.button('help','Como usar',(1095,75,130,40),self.show_help)
        self.button('open','Abrir projeto',(1235,75,154,40),self.open_request)
        self.button('save','Salvar projeto',(1399,75,177,40),self.save_file)
        self.text('MODELO',28,104,'small',MUTED)
        self.text('2 entradas  →  classificadores lineares  →  softmax',100,101,'bold',TEXT)
        if self.running:
            self.text('● TREINANDO',670,104,'small',GREEN)
        elif self.history:
            self.text('● MODELO PRONTO',670,104,'small',CYAN)
        else:
            self.text('● AGUARDANDO TREINAMENTO',670,104,'small',MUTED)

    def footer(self):
        pygame.draw.line(self.canvas,BORDER,(26,850),(1574,850))
        self.text('STATUS',27,865,'small',CYAN)
        self.text(self.status,94,863,'body',TEXT, max_width=1450)

    def section(self,number,title,x,y):
        self.text(number,x,y,'head',CYAN)
        self.text(title,x+41,y,'head')

    def world_to_screen(self,x,y):
        return (int(self.plot.left+(x+10)/20*self.plot.width),
                int(self.plot.bottom-(y+10)/20*self.plot.height))

    def screen_to_world(self,x,y):
        x1=-10+20*(x-self.plot.left)/self.plot.width
        x2=-10+20*(self.plot.bottom-y)/self.plot.height
        return round(max(-10,min(10,x1)),2), round(max(-10,min(10,x2)),2)

    def update_region_surface(self):
        if not self.net or not self.history or not self.show_regions:
            return None
        key=(self.model_version, tuple(self.net.classes))
        if self.cache_key==key:
            return self.region_cache
        rw,rh=220,136
        x1=np.linspace(-10,10,rw)
        x2=np.linspace(10,-10,rh)
        xx,yy=np.meshgrid(x1,x2,indexing='ij')
        coords=np.column_stack((xx.ravel(),yy.ravel()))
        indices=np.argmax(self.net.probabilidades(coords),axis=1).reshape(rw,rh)
        pixels=np.empty((rw,rh,3),dtype=np.uint8)
        for index,klass in enumerate(self.net.classes):
            pixels[indices==index]=blend((15,18,23),CLASS_COLORS[klass],.20)
        surf=pygame.surfarray.make_surface(pixels)
        self.region_cache=pygame.transform.scale(surf,self.plot.size)
        self.cache_key=key
        return self.region_cache

    def draw_graph(self):
        self.panel((24,143,850,691))
        self.section('01','GRÁFICO DE CLASSIFICAÇÃO',43,161)
        counts=contagens(self.points)
        self.text('Pontos rotulados e regiões previstas pelo modelo',43,195,'body',MUTED)
        self.text(f"{len(self.points)} pontos  •  {sum(v>0 for v in counts.values())} classes",626,198,'small',MUTED)
        self.text('Contorno vermelho: erro no treinamento', 453, 220, 'tiny', RED)
        self.panel(self.plot,DARK,(76,84,95),4)
        region=self.update_region_surface()
        if region is not None:
            self.canvas.blit(region,self.plot.topleft)
        # Grid and numeric axes, with equal coordinate ranges on both dimensions.
        for tick in range(-10,11,2):
            x,_=self.world_to_screen(tick,0)
            _,y=self.world_to_screen(0,tick)
            pygame.draw.line(self.canvas,(52,59,70),(x,self.plot.top),(x,self.plot.bottom),1)
            pygame.draw.line(self.canvas,(52,59,70),(self.plot.left,y),(self.plot.right,y),1)
            self.center(str(tick),(x-15,self.plot.bottom+8,30,20),'tiny',MUTED)
            self.center(str(tick),(self.plot.left-36,y-9,30,18),'tiny',MUTED)
        x0,y0=self.world_to_screen(0,0)
        pygame.draw.line(self.canvas,(139,149,161),(x0,self.plot.top),(x0,self.plot.bottom),1)
        pygame.draw.line(self.canvas,(139,149,161),(self.plot.left,y0),(self.plot.right,y0),1)
        if region is not None:
            old_clip=self.canvas.get_clip()
            self.canvas.set_clip(self.plot)
            for _,_,p0,p1 in segmentos_decisao(self.net):
                pygame.draw.aaline(self.canvas,(196,210,221),self.world_to_screen(*p0),self.world_to_screen(*p1))
            self.canvas.set_clip(old_clip)
        predicted=[]
        if self.net and self.history and self.points:
            coords=np.asarray([[p['x1'],p['x2']] for p in self.points])
            predicted=[self.net.classes[int(j)] for j in np.argmax(self.net.probabilidades(coords),axis=1)]
        for idx,p in enumerate(self.points):
            x,y=self.world_to_screen(p['x1'],p['x2'])
            wrong=bool(predicted) and predicted[idx]!=p['classe']
            pygame.draw.circle(self.canvas, RED if wrong else (4,7,10), (x,y), 9 if wrong else 8)
            pygame.draw.circle(self.canvas,CLASS_COLORS[p['classe']],(x,y),6)
            pygame.draw.circle(self.canvas,(230,238,244),(x,y),6,1)
        if self.test_point is not None:
            x,y=self.world_to_screen(*self.test_point)
            poly=[(x,y-11),(x+11,y),(x,y+11),(x-11,y)]
            pygame.draw.polygon(self.canvas,(13,16,22),poly)
            pygame.draw.polygon(self.canvas,GOLD,poly,3)
            pygame.draw.line(self.canvas,GOLD,(x-3,y),(x+3,y),2)
            pygame.draw.line(self.canvas,GOLD,(x,y-3),(x,y+3),2)
        pygame.draw.rect(self.canvas,(116,126,138),self.plot,1)
        self.text('CARACTERÍSTICA 1  (x1)',320,735,'small',MUTED)
        self.text('x2',43,225,'bold',MUTED)
        for i,c in enumerate(ROTULOS):
            x=51+i*125
            pygame.draw.circle(self.canvas,CLASS_COLORS[c],(x,788),6)
            self.text(f'Classe {c}: {counts[c]}',x+13,779,'small',TEXT)
        self.button('regions','Regiões: '+('LIGADAS' if self.show_regions else 'DESLIGADAS'),
                    (464,771,190,37),self.toggle_regions,'active' if self.show_regions else 'normal')
        self.text('◆ ponto de teste',682,780,'small',GOLD)

    def draw_data_controls(self):
        self.panel((890,143,330,691))
        self.section('02','DADOS E TREINO',909,161)
        self.text('Selecione a classe dos pontos',910,194,'small',MUTED)
        counts=contagens(self.points)
        for i,c in enumerate(ROTULOS):
            self.button('class_'+c,f'{c}   ({counts[c]})',(909+99*i,218,90,36),
                        lambda c=c:self.choose_class(c),'active' if c==self.selected_class else 'normal',not self.running,'bold')
            pygame.draw.circle(self.canvas,CLASS_COLORS[c],(916+99*i,236),3)
        self.text('Clique no gráfico para:',910,270,'small',MUTED)
        self.button('add_mode','Adicionar pontos',(909,294,145,37),lambda:self.set_mode('adicionar'),
                    'active' if self.mode=='adicionar' else 'normal',not self.running)
        self.button('predict_mode','Classificar ponto',(1062,294,140,37),lambda:self.set_mode('prever'),
                    'active' if self.mode=='prever' else 'normal')
        self.text('Ou informe as coordenadas:',910,346,'small',MUTED)
        for i,key in enumerate(('x1','x2')):
            x=909+i*92
            self.text(key.upper(),x,371,'small',MUTED)
            self.input_field(key,(x,394,83,34))
        self.button('coord_action','Prever' if self.mode=='prever' else 'Adicionar',
                    (1098,394,104,34),self.coordinate_action,'primary', self.mode=='prever' or not self.running)
        self.text('EXEMPLOS PRONTOS',910,446,'small',MUTED)
        self.button('two','2 classes',(909,469,93,33),lambda:self.example_request('binario'),enabled=not self.running)
        self.button('three','3 classes',(1009,469,93,33),lambda:self.example_request('tres'),enabled=not self.running)
        self.button('xor','XOR',(1109,469,93,33),lambda:self.example_request('xor'),enabled=not self.running)
        self.button('clear','Limpar pontos',(909,510,293,34),self.clear_request,'danger',bool(self.points) and not self.running)
        pygame.draw.line(self.canvas,BORDER,(908,560),(1202,560))
        self.text('TREINAMENTO',910,574,'bold',TEXT)
        self.option_row('Taxa η',f'{self.rate:.2f}','rate',606)
        self.option_row('Limite de passos',str(self.max_steps),'limit',646)
        self.option_row('Intervalo',f'{self.interval:.2f} s','interval',686)
        self.button('step','Passo',(909,735,90,37),self.step,enabled=not self.running)
        self.button('start','Treinar',(1007,735,93,37),self.start,'primary',not self.running)
        self.button('fast','Rápido',(1108,735,94,37),self.start_fast,'normal',not self.running)
        self.button('pause','Pausar',(909,781,142,36),self.pause,enabled=self.running)
        self.button('reset','Reiniciar modelo',(1059,781,143,36),self.reset,enabled=bool(self.net) and not self.running)

    def input_field(self,key,rect):
        rect=pygame.Rect(rect)
        self.panel(rect,DARK,CYAN if self.focus==key else BORDER,5)
        self.text(self.coord[key] or ('_' if self.focus==key else ''),rect.x+7,rect.y+7,'mono',TEXT,max_width=rect.w-12)
        self.buttons['input_'+key]=(rect,lambda:self.focus_field(key),True)

    def option_row(self,label,val,key,y):
        self.text(label,910,y+7,'small',MUTED)
        self.button(key+'_minus','−',(1062,y,32,31),lambda:self.change_option(key,-1),enabled=not self.running,style='bold')
        self.panel((1098,y,65,31),DARK,BORDER,5)
        self.center(val,(1098,y,65,31),'small',CYAN)
        self.button(key+'_plus','+',(1167,y,34,31),lambda:self.change_option(key,1),enabled=not self.running,style='bold')

    def draw_monitor(self):
        self.panel((1236,143,340,691))
        self.section('03','ACOMPANHAMENTO',1255,161)
        self.text('Métricas na base de treinamento',1255,194,'small',MUTED)
        e=self.history[-1] if self.history else None
        data=[('PASSO',str(len(self.history))),('PERDA',f"{e['perda']:.4f}" if e else '—'),
              ('ACURÁCIA',f"{e['acuracia']*100:.1f}%" if e else '—')]
        for i,(name,value) in enumerate(data):
            x=1255+i*100
            self.panel((x,218,92,68),DARK,BORDER,7)
            self.text(name,x+8,228,'tiny',MUTED)
            self.text(value,x+8,249,'bold',GREEN if i==2 else TEXT,max_width=78)
        self.panel((1255,304,302,169),DARK,BORDER,8)
        self.text('EVOLUÇÃO DA PERDA',1268,317,'small',MUTED)
        self.loss_chart(pygame.Rect(1279,349,253,92))
        self.text('Entropia cruzada média',1269,451,'tiny',CYAN)
        self.text('MODELO E PARÂMETROS',1255,493,'bold')
        self.text('z = w1·(x1/10) + w2·(x2/10) + b',1255,521,'small',MUTED)
        self.text('p(classe) = softmax(z)',1255,543,'small',MUTED)
        self.text('CLASSE       w1       w2         b',1255,574,'tiny',DIM)
        for row,c in enumerate(ROTULOS):
            y=595+row*24
            if self.net and c in self.net.classes:
                i=self.net.classes.index(c)
                w1,w2=self.net.W[:,i];b=self.net.b[i]
                self.text(c,1258,y,'mono',CLASS_COLORS[c])
                self.text(f'{w1:7.3f} {w2:8.3f} {b:9.3f}',1293,y,'mono',TEXT,max_width=255)
            else:
                self.text(f'{c}       —        —          —',1258,y,'mono',DIM)
        pygame.draw.line(self.canvas,BORDER,(1255,681),(1556,681))
        self.text('PONTO DE TESTE',1255,694,'bold')
        if self.test_point is None:
            self.text('Selecione “Classificar ponto” e',1255,724,'small',MUTED)
            self.text('clique em qualquer lugar do gráfico.',1255,746,'small',MUTED)
        elif not self.history:
            self.text(f'x1={self.test_point[0]:.2f}  |  x2={self.test_point[1]:.2f}',1255,723,'small',TEXT)
            self.text('Treine o modelo para obter uma resposta.',1255,750,'small',MUTED)
        else:
            probs=self.net.probabilidades(self.test_point)[0]
            top=int(np.argmax(probs)); label=self.net.classes[top]
            self.text(f'x1={self.test_point[0]:.2f}  |  x2={self.test_point[1]:.2f}',1255,720,'small',MUTED)
            self.text(f'Previsto: {label}  ({probs[top]*100:.1f}%)',1255,746,'bold',CLASS_COLORS[label])
            info='   '.join(f'{c}: {p*100:.0f}%' for c,p in zip(self.net.classes,probs))
            self.text(info,1255,776,'small',TEXT,max_width=295)
            self.text('Percentuais relativos; não garantem acerto.',1255,802,'tiny',DIM)

    def loss_chart(self,rect):
        for i in range(4):
            y=rect.top+i*rect.height//3
            pygame.draw.line(self.canvas,(45,53,64),(rect.left,y),(rect.right,y),1)
        if len(self.history)<2:
            self.center('Gráfico após dois passos',(rect.left,rect.top+30,rect.width,25),'tiny',DIM)
            return
        values=np.asarray([e['perda'] for e in self.history])
        indexes=np.unique(np.linspace(0,len(values)-1,min(rect.width,len(values)),dtype=int))
        maximum=max(float(values.max()),1e-12)
        xy=[(rect.x+int(i/(len(values)-1)*rect.width),rect.bottom-int(values[i]/maximum*rect.height)) for i in indexes]
        pygame.draw.lines(self.canvas,CYAN,False,xy,2)

    def toggle_regions(self):
        self.show_regions=not self.show_regions
        self.status='Regiões de decisão '+('ativadas.' if self.show_regions else 'ocultadas.')

    def choose_class(self,c):
        self.selected_class=c
        self.status=f'Classe {c} selecionada para novos pontos.'

    def set_mode(self,mode):
        self.mode=mode
        if mode=='prever':
            self.status='Clique no gráfico ou digite coordenadas para classificar um ponto de teste.'
        else:
            self.status=f'Clique no gráfico para adicionar pontos da classe {self.selected_class}.'

    def focus_field(self,key):
        self.focus=key
        self.select_all=True
        self.status=f'Informe {key.upper()} entre -10 e 10. Use ponto ou vírgula decimal.'

    def clear_model(self):
        self.running=False
        self.net=None
        self.X=self.y=None
        self.history=[]
        self.model_version+=1
        self.cache_key=None
        self.region_cache=None

    def invalidate_data(self):
        had_model=bool(self.net)
        self.clear_model()
        return had_model

    def add_point(self,x,y):
        if self.running:
            self.status='Pause o treinamento antes de alterar os pontos.'
            return
        if len(self.points)>=LIMITE:
            self.status=f'Limite de {LIMITE} pontos atingido.'
            return
        try:
            p=validar_ponto({'x1':x,'x2':y,'classe':self.selected_class})
        except (KeyError,ValueError,TypeError) as exc:
            self.status=str(exc)
            return
        had=self.invalidate_data()
        conflicting=any(z['x1']==p['x1'] and z['x2']==p['x2'] and z['classe']!=p['classe'] for z in self.points)
        self.points.append(p)
        self.status=f"Ponto ({x:.2f}, {y:.2f}) adicionado à classe {self.selected_class}."
        if had: self.status+=' A base mudou; treine o modelo novamente.'
        if conflicting: self.status+=' Atenção: coordenadas idênticas têm classes diferentes.'

    def coordinate_action(self):
        self.focus=None
        try:
            x=float(self.coord['x1'].replace(',','.'))
            y=float(self.coord['x2'].replace(',','.'))
            if not all(math.isfinite(v) and -10 <= v <= 10 for v in (x,y)):
                raise ValueError
        except ValueError:
            self.status='Coordenadas inválidas: informe valores entre -10 e 10.'
            return
        if self.mode=='prever': self.set_test(x,y)
        else: self.add_point(x,y)

    def set_test(self,x,y):
        self.test_point=(float(x),float(y))
        self.coord={'x1':f'{x:.2f}', 'x2':f'{y:.2f}'}
        self.status='Ponto de teste atualizado. Ele não foi adicionado à base de treinamento.'

    def remove_nearest(self,pos):
        if self.running:
            self.status='Pause o treinamento antes de remover pontos.'
            return
        if not self.points: return
        distances=[]
        for p in self.points:
            px,py=self.world_to_screen(p['x1'],p['x2'])
            distances.append((px-pos[0])**2+(py-pos[1])**2)
        idx=int(np.argmin(distances))
        if distances[idx] > 15**2:
            self.status='Clique com o botão direito perto de um ponto para removê-lo.'
            return
        removed=self.points.pop(idx)
        self.invalidate_data()
        self.status=f"Ponto da classe {removed['classe']} removido; o modelo foi reiniciado."

    def example_request(self,kind):
        if self.points:
            self.modal=('Substituir os pontos?',
                        ['O exemplo escolhido substituirá a base e reiniciará o modelo.',
                         'Se desejar preservar os pontos atuais, salve o projeto primeiro.'],
                        lambda:self.load_example(kind))
        else:
            self.load_example(kind)

    def load_example(self,kind):
        self.points=exemplo(kind)
        self.test_point=None
        self.clear_model()
        self.mode='adicionar'
        self.selected_class='A'
        if kind=='xor':
            self.status='Exemplo XOR: uma fronteira linear não separa perfeitamente as duas classes.'
        else:
            self.status=f"Exemplo com {'duas' if kind=='binario' else 'três'} classes carregado. Treine para ver as regiões."

    def clear_request(self):
        self.modal=('Limpar todos os pontos?',
                    ['A base atual e o modelo treinado serão removidos.',
                     'O projeto pode ser salvo antes de continuar.'],self.clear_points)

    def clear_points(self):
        self.points=[]
        self.test_point=None
        self.clear_model()
        self.status='Base limpa. Escolha uma classe e adicione pontos ao gráfico.'

    def change_option(self,key,amount):
        name,choices={'rate':('rate',self.rates),'limit':('max_steps',self.max_steps_options),
                      'interval':('interval',self.time_options)}[key]
        current=getattr(self,name)
        index=choices.index(current)
        setattr(self,name,choices[max(0,min(len(choices)-1,index+amount))])
        self.status='Configuração atualizada.'

    def ensure_model(self):
        classes=sorted({p['classe'] for p in self.points})
        if len(classes)<2:
            self.status='Cadastre pontos de pelo menos duas classes antes de treinar.'
            return False
        if self.net is None:
            self.X,self.y,classes=dados_para_matrizes(self.points)
            self.net=Classificador(classes)
            self.history=[]
            self.model_version+=1
        if self.X is None:
            self.X,self.y,_=dados_para_matrizes(self.points,self.net.classes)
        return True

    def step(self):
        if self.running or not self.ensure_model(): return
        if len(self.history)>=self.max_steps:
            self.status='Limite alcançado: aumente o limite ou reinicie o modelo.'
            return
        self.advance(1)

    def start(self):
        self._start(False)

    def start_fast(self):
        self._start(True)

    def _start(self,fast):
        if not self.ensure_model(): return
        if len(self.history)>=self.max_steps:
            self.status='Limite alcançado: aumente o limite ou reinicie o modelo.'
            return
        self.fast=fast
        self.running=True
        self.last_tick=time.monotonic()-self.interval
        self.status='Treinamento em andamento: a fronteira acompanha os parâmetros.'

    def pause(self):
        self.running=False
        self.status=f'Treinamento pausado após {len(self.history)} passo(s).'

    def reset(self):
        self.clear_model()
        self.status='Modelo reiniciado. Os pontos e as configurações foram preservados.'

    def advance(self,n):
        for _ in range(n):
            if len(self.history)>=self.max_steps: break
            loss,acc,hits=self.net.passo(self.X,self.y,self.rate)
            self.history.append({'passo':len(self.history)+1,'perda':loss,'acuracia':acc,'acertos':hits})
        self.model_version+=1
        if len(self.history)>=self.max_steps:
            self.running=False
            self.status=f'Treinamento terminou em {len(self.history)} passos; experimente classificar pontos.'
        elif not self.running:
            self.status=f'Passo {len(self.history)} concluído. Observe a perda e a fronteira.'

    def show_help(self):
        self.modal=('Como usar',[
            '1. Comece com o exemplo ou escolha “2 classes”, “3 classes” ou “XOR”.',
            '2. Clique em Passo, Treinar ou Rápido; observe a perda e as regiões.',
            '3. Em “Classificar ponto”, clique no gráfico para testar um dado novo.',
            '4. Para criar sua base, selecione a classe e clique em “Adicionar pontos”.',
            '5. O botão direito do mouse remove o ponto de treinamento mais próximo.',
            'A acurácia apresentada usa apenas os pontos da base de treinamento.',
            'A regressão logística multiclasses cria regiões com fronteiras lineares.',
        ],None)

    def open_request(self):
        if self.points:
            self.modal=('Abrir outro projeto?', ['A base, o modelo e o histórico atuais serão substituídos.',
                        'Salve o projeto atual antes, se necessário.'],self.open_file)
        else:self.open_file()

    def file_dialog(self,save_mode):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
            try:
                kwargs={'title':'Salvar projeto' if save_mode else 'Abrir projeto',
                        'filetypes':[('Projeto de classificação','*.json'),('Todos os arquivos','*.*')]}
                if save_mode:
                    kwargs.update(defaultextension='.json',initialfile='projeto_classificacao.json')
                    return filedialog.asksaveasfilename(**kwargs)
                return filedialog.askopenfilename(**kwargs)
            finally:root.destroy()
        except Exception as exc:
            self.status=f'Não foi possível abrir o seletor de arquivos: {exc}'
            return ''

    def save_file(self):
        path=self.file_dialog(True)
        if not path:return
        cfg={'taxa':self.rate,'limite':self.max_steps,'intervalo':self.interval,
             'classe':self.selected_class,'modo':self.mode,'teste':self.test_point,'regioes':self.show_regions}
        try:
            salvar(path,self.points,self.net,self.history,cfg)
            self.status=f'Projeto salvo: {Path(path).name}'
        except (OSError,ValueError,TypeError) as exc:
            self.status=f'Não foi possível salvar: {exc}'

    def open_file(self):
        path=self.file_dialog(False)
        if not path:return
        try:
            points,net,history,cfg=carregar(path)
            rate=float(cfg.get('taxa',.5));limit=int(cfg.get('limite',500));interval=float(cfg.get('intervalo',.1))
            label=cfg.get('classe','A');mode=cfg.get('modo','adicionar');test=cfg.get('teste')
            if rate not in self.rates or limit not in self.max_steps_options or interval not in self.time_options or label not in ROTULOS or mode not in ('adicionar','prever'):
                raise ValueError('Configuração incompatível.')
            if test is not None:
                if not isinstance(test,(list,tuple)) or len(test)!=2: raise ValueError('Ponto de teste inválido.')
                validated=validar_ponto({'x1':test[0],'x2':test[1],'classe':'A'})
                test=(validated['x1'],validated['x2'])
            X,y=(None,None)
            if net:X,y,_=dados_para_matrizes(points,net.classes)
            self.running=False
            self.points,self.net,self.history=points,net,history
            self.rate,self.max_steps,self.interval=rate,limit,interval
            self.selected_class,self.mode,self.test_point=label,mode,test
            self.show_regions=bool(cfg.get('regioes',True))
            self.X,self.y=X,y
            self.model_version+=1
            self.cache_key=None
            self.status=f'Projeto carregado: {Path(path).name}'
        except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:
            self.status=f'Não foi possível abrir: {exc}'

    def close_modal(self):self.modal=None

    def confirm_modal(self):
        action=self.modal[2]
        self.modal=None
        if action:action()

    def draw_modal(self):
        title,lines,action=self.modal
        shade=pygame.Surface((W,H),pygame.SRCALPHA);shade.fill((0,0,0,180))
        self.canvas.blit(shade,(0,0))
        height=165+len(lines)*28
        box=pygame.Rect(350,(H-height)//2,900,height)
        self.panel(box,(25,30,40),(87,105,121),12)
        self.text(title,box.x+28,box.y+25,'head',CYAN)
        for i,line in enumerate(lines):self.text(line,box.x+28,box.y+65+i*28,'body',TEXT,max_width=842)
        self.buttons={}
        if action is None:
            self.button('modal_close','Entendi',(box.right-170,box.bottom-62,140,40),self.close_modal,'primary')
        else:
            self.button('modal_no','Cancelar',(box.right-318,box.bottom-62,132,40),self.close_modal)
            self.button('modal_yes','Continuar',(box.right-170,box.bottom-62,140,40),self.confirm_modal,'primary')

    def render(self):
        self.buttons={}
        self.canvas.fill(BG)
        self.header();self.draw_graph();self.draw_data_controls();self.draw_monitor();self.footer()
        if self.modal:self.draw_modal()
        ww,wh=self.window.get_size();s=min(ww/W,wh/H)
        dw,dh=max(1,int(W*s)),max(1,int(H*s));x=(ww-dw)//2;y=(wh-dh)//2
        self.viewport=(x,y,s)
        self.window.fill((5,6,9))
        self.window.blit(self.canvas if (dw,dh)==(W,H) else pygame.transform.smoothscale(self.canvas,(dw,dh)),(x,y))
        pygame.display.flip()

    def handle_event(self,event):
        if event.type==pygame.QUIT:
            self.alive=False;return
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE:
                if self.modal:self.close_modal()
                else:self.focus=None
                return
            if self.modal:return
            if self.focus:
                if event.key in (pygame.K_RETURN,pygame.K_KP_ENTER):
                    self.coordinate_action();return
                if event.key==pygame.K_TAB:
                    self.focus='x2' if self.focus=='x1' else 'x1';self.select_all=True;return
                if event.key in (pygame.K_BACKSPACE,pygame.K_DELETE):
                    self.coord[self.focus]='' if self.select_all else self.coord[self.focus][:-1]
                    self.select_all=False;return
                c=getattr(event,'unicode','')
                if c in '0123456789.,-' and c and len(self.coord[self.focus])<14:
                    self.coord[self.focus]=c if self.select_all else self.coord[self.focus]+c
                    self.select_all=False
                return
            if event.key==pygame.K_SPACE:self.step()
            elif event.key==pygame.K_RETURN:self.pause() if self.running else self.start()
            return
        if event.type==pygame.MOUSEBUTTONDOWN and event.button in (1,3):
            pos=self.logical(event.pos)
            if self.modal:
                if event.button==1:
                    for rect,action,enabled in list(self.buttons.values()):
                        if enabled and rect.collidepoint(pos):action();break
                return
            if event.button==1:
                for name,(rect,action,enabled) in list(self.buttons.items()):
                    if enabled and rect.collidepoint(pos):
                        if not name.startswith('input_'):self.focus=None
                        action();return
                self.focus=None
                if self.plot.collidepoint(pos):
                    x,y=self.screen_to_world(*pos)
                    if self.mode=='prever':self.set_test(x,y)
                    else:self.add_point(x,y)
            elif self.plot.collidepoint(pos):self.remove_nearest(pos)

    def run(self):
        while self.alive:
            self.render()
            for event in pygame.event.get():self.handle_event(event)
            if self.running and self.modal is None:
                now=time.monotonic()
                if self.fast:
                    self.advance(20)
                elif now-self.last_tick>=self.interval:
                    self.advance(1);self.last_tick=now
            self.clock.tick(FPS)
        pygame.quit()


if __name__=='__main__':
    App().run()
