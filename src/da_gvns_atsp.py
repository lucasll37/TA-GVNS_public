import os
import re
import uuid
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsplib95
import networkx as nx

from typing import List, Tuple, Dict, Optional, Union
from matplotlib.lines import Line2D

# recria o alias que o NetworkX espera
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all

class DA_GVNS:
    frame_number = 0
    
    def __init__(self, path: str, first_node: Optional[int]=None, benchmark_files: Optional[List[str]]=None) -> None:
        self.name_problem = path.split('/')[-1].split('.')[0]
        self.n, self.cost_matrix, self.G = self._load_tsplib_instance(path)
        self.pos = nx.spring_layout(self.G, seed=42)
        self.best = self._nearest_neighbor_init(first_node)
        self.best_cost = self._evaluate(self.best)
        self.init_order: List[str] = ['relocate', 'swap', '2opt']
        self.benchmark = self._load_benchmark(self.name_problem, benchmark_files)
        self.to_print = False
        self.start = None
        self.time_limit = None
        self.steps = 0
        
    @classmethod
    def _get_fn(cls):
        frame_number = cls.frame_number
        cls.frame_number += 1
        return frame_number
    
    @classmethod
    def _reset_fn(cls):
        cls.frame_number = 0
        
    def _load_tsplib_instance(self, path: str) -> Tuple[int, np.ndarray, nx.Graph]:
        problem = tsplib95.load(path)
        n = problem.dimension
        cost_matrix = np.zeros((n, n), dtype=float)
        G = problem.get_graph()
        G.remove_edges_from(nx.selfloop_edges(G))
        
        for i, j in problem.get_edges():
            cost_matrix[i-1, j-1] = problem.get_weight(i, j)
            
        return n, cost_matrix, G
    
    def _nearest_neighbor_init(self, first_node: Optional[int]=None) -> np.ndarray:
        unvisited = set(range(0, self.n))
        
        if first_node is None:
            first_node = unvisited.pop()
        else:
            unvisited.remove(first_node)
            
        tour = np.array([first_node], dtype=int)
        while len(unvisited) > 0:
            current = tour[-1]
            next_node = min(unvisited, key=lambda j: self.cost_matrix[current][j])
            tour = np.append(tour, next_node)
            unvisited.remove(next_node)
        
        tour = np.append(tour, first_node)
        
        return tour
    
    def _load_benchmark(self, name_problem: str, paths: List[str]) -> Optional[int]:
        if paths is None:
            return None
        
        # padrão genérico: captura chave e valor
        pattern = re.compile(r'^(\S+)\s*:\s*(\d+)')
        
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    m = pattern.match(line)
                    if not m:
                        continue
                    key, val = m.group(1), int(m.group(2))
                    if key == name_problem:
                        return val
        # não achou em nenhum arquivo
        return None
    
    def _evaluate(self, tour: np.ndarray) -> float:
        return float(sum(self.cost_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)))
    
    def _shake(self, tour: np.ndarray, operator:str) -> np.ndarray:
        self.steps += 1
        new = tour[:-1].copy()
        i, j = sorted(random.sample(range(0, self.n), 2))
        
        if operator == 'relocate':
            node = new[i]
            new = np.delete(new, i)
            new = np.append(new, node)
            
        elif operator == 'swap':
            new[i], new[j] = new[j], new[i]
            
        elif operator == '2opt':
            new[i:j+1] = new[i:j+1][::-1]
            
        else:
            raise ValueError(f"Operador desconhecido: {operator}")

        new = np.append(new, new[0])
        
        return new
    
    def _local_search_best_move(self, tour: np.ndarray, operator: str) -> Tuple[np.ndarray, float]:
        best = tour
        cost0 = self._evaluate(tour)
        best_delta = 0.0

        for i in range(1, self.n - 1):
            
            if time.time()-self.start > self.time_limit:
                break
            
            for j in range(i+1, self.n - 1):
                self.steps += 1
                candidate = tour[:-1].copy()
                
                if operator == 'relocate':
                    node = candidate[i]
                    candidate = np.delete(candidate, i)
                    candidate = np.append(candidate, node)
            
                elif operator == 'swap':
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    
                elif operator == '2opt':
                    candidate[i:j+1] = candidate[i:j+1][::-1]
                    
                candidate = np.append(candidate, candidate[0])
                delta = self._evaluate(candidate) - cost0
                
                if delta < best_delta:
                    best_delta = delta
                    best = candidate.copy()
                    
                if operator == 'relocate':
                    break
                    
        return best, best_delta
    
    def _sequential_vnd(self, tour: np.ndarray, local_search_order: List[str], local_search_counters: Dict[str,int]) -> Tuple[np.ndarray, bool]:
        improved_any = False
        
        for op in local_search_order:
            improved = True
            
            while improved:
                best, delta = self._local_search_best_move(tour, op)
                if delta < 0:
                    tour = best
                    local_search_counters[op] += 1
                    improved_any = True
                    
                    if self.to_print:
                        self.plot_graph(tour=tour, color_tour='red', to_video=True)
                else:
                    improved = False
                    
        return tour, improved_any

    def _adapt_order(self, curr_order: List[str], counters: Dict[str,int], no_success: bool) -> List[str]:
        if no_success:
            return self.init_order.copy()
        
        return sorted(curr_order, key=lambda op: counters[op], reverse=True)
    
    
    def search(self, k_max: int, time_limit: float=60.0, seed: Optional[int]=None, to_print: bool=False, 
            memory_size: int=10) -> Tuple[np.ndarray, float, Optional[float], int, pd.DataFrame]:
        """
        Executa a busca DA-GVNS com memória adaptativa para seleção de métodos VND
        
        Args:
            k_max: Número máximo de vizinhanças
            time_limit: Limite de tempo para execução
            seed: Semente para reprodutibilidade
            to_print: Se deve gerar visualizações
            memory_size: Tamanho da memória para cada método VND
            
        Returns:
            Tour, custo, tempo de busca, número de passos, DataFrame com métricas
        """
        if seed is not None:
            random.seed(seed)
            
        self.to_print = to_print
        self.time_limit = time_limit
        self.__class__._reset_fn()

        # Inicializar dataframe para registro
        metrics_data = {
            'iteration': [],
            'time': [],
            'steps': [],
            'cost': [],
            'best_cost': [],
            'benchmark': [],
            'shake_operator': [],
            'improvement': []
        }
                
        iteration_count = 0
        shake_ops = self.init_order.copy()
        local_search_ops = self.init_order.copy()
        
        shake_counters = {op:0 for op in shake_ops}
        local_search_counters = {op:0 for op in local_search_ops}
        
        shake_any_improv = False
        local_search_any_improv = False
        
        metrics_entry = {
            'iteration': 0,
            'time': 0,
            'steps': self.steps,
            'cost': self.best_cost,
            'best_cost': self.best_cost,
            'benchmark': self.benchmark if self.benchmark is not None else float('nan'),
            'vnd_method': "",
            'shake_operator': "",
            'improvement': 0
        }
                
        self.start = time.time()
        
        while time.time()-self.start < self.time_limit:
            iteration_count += 1
            
            # Adaptar a ordem dos operadores de shake
            shake_ops = self._adapt_order(shake_ops, shake_counters, not shake_any_improv)
            shake_any_improv = False
            
            if self.to_print:
                self.plot_graph(tour=self.best, color_tour='black', to_video=True)
                
            for k in range(k_max):
                if time.time()-self.start > self.time_limit:
                    break
                
                op = shake_ops[k%len(shake_ops)]
                s_star = self._shake(self.best, op)
                
                if self.to_print:
                    self.plot_graph(tour=s_star, color_tour='red', to_video=True)
                
                # Adaptar a ordem dos operadores de busca local
                local_search_ops = self._adapt_order(local_search_ops, local_search_counters, not local_search_any_improv)
                local_search_any_improv = False
                
                # Aplicar o método VND selecionado
                old_step = self.steps
                s_new, improved_ls = self._sequential_vnd(s_star, local_search_ops, local_search_counters)
                
                if improved_ls:
                    local_search_any_improv = True
                    
                # Avaliar a nova solução
                new_cost = self._evaluate(s_new)
                cost_update = new_cost - self.best_cost
                cost_update_weighted = cost_update / (self.steps - old_step)
                
                # Verificar se encontramos uma solução melhor
                if cost_update < 0:
                    self.best, self.best_cost = s_new, new_cost
                    shake_counters[op] += 1
                    shake_any_improv = True
                    
                    if self.to_print:
                        self.plot_graph(tour=self.best, color_tour='black', to_video=True)
                        
                # Registro básico
                metrics_entry = {
                    'iteration': iteration_count,
                    'time': time.time() - self.start,
                    'steps': self.steps,
                    'cost': new_cost,
                    'best_cost': self.best_cost,
                    'benchmark': self.benchmark if self.benchmark is not None else float('nan'),
                    'shake_operator': op,
                    'improvement': cost_update
                }
                
                # Adicionar entrada ao dataframe
                for key, value in metrics_entry.items():
                    metrics_data[key].append(value)
                    
                # Verificar se encontramos a solução ótima
                if self.benchmark is not None and self.best_cost == self.benchmark:
                    history = {
                        'best': self.best,
                        'best_cost': self.best_cost,
                        'search_time': time.time()-self.start ,
                        'steps': self.steps,
                        'metrics_df': pd.DataFrame(metrics_data)
                    }
                    
                    return history
        
        history = {
            'best': self.best,
            'best_cost': self.best_cost,
            'search_time': time.time()-self.start ,
            'steps': self.steps,
            'metrics_df': pd.DataFrame(metrics_data)
        }
        
        return history
                        
    def plot_graph(self, tour: Optional[Union[np.ndarray, str]]=None, color_tour: str = 'red', to_video: bool=False) -> None:
        edge_weights = [self.G[u][v].get('weight', 1) for u, v in self.G.edges()]
        cmap_edges = plt.cm.coolwarm

        plt.figure(figsize=(8, 8))
        
        if self.benchmark is not None:
            benchmark = self.benchmark
        else:
            benchmark = '-'
            
        if isinstance(tour, str) and tour == 'best':
            cost = self._evaluate(self.best)
            plt.title(f'Graph TSP/ATSP: {self.name_problem} [TSPLIB]\n(cost: {cost:.1f}/bench: {benchmark})', fontsize=16, fontweight='bold')
            
        elif isinstance(tour, np.ndarray):
            cost = self._evaluate(tour)
            plt.title(f'Graph TSP/ATSP: {self.name_problem} [TSPLIB]\n(cost: {cost:.1f}/bench: {benchmark})', fontsize=16, fontweight='bold')
            
        else:
            plt.title(f'Graph TSP/ATSP: {self.name_problem} [TSPLIB]\n(cost: -/bench: {benchmark})', fontsize=16, fontweight='bold')
        
        nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color='black',
            node_size=300,
            alpha=1,
            linewidths=0.5,
            edgecolors='black'
        )

        nx.draw_networkx_labels(
            self.G, self.pos,
            font_size=8,
            font_color='white',
            font_weight='bold'
        )
        
        nx.draw_networkx_edges(
            self.G, self.pos,
            width=1,
            edge_color=edge_weights,
            edge_cmap=cmap_edges,
            style='solid',
            alpha=0.6,
            arrows=True,
            connectionstyle='arc3,rad=0.1'
        )
        
        path_edges = None
        
        # if tour is not None and isinstance(tour, np.ndarray):
        #     path_edges = [(tour[i] + 1, tour[i+1] + 1) for i in range(len(tour) - 1)]
            
        # elif isinstance(tour, str) and tour == 'best':
        #     path_edges = [(self.best[i] + 1, self.best[i+1] + 1) for i in range(len(self.best) - 1)]
        
        if tour is not None and isinstance(tour, np.ndarray):
            # Obter a lista de nós reais do grafo
            node_list = list(self.G.nodes())
            # Mapear os índices do tour para os nós reais do grafo
            path_edges = []
            for i in range(len(tour) - 1):
                # Verificar se os índices estão dentro dos limites
                if tour[i] < len(node_list) and tour[i+1] < len(node_list):
                    path_edges.append((node_list[tour[i]], node_list[tour[i+1]]))
                    
        elif isinstance(tour, str) and tour == 'best':
            # Obter a lista de nós reais do grafo
            node_list = list(self.G.nodes())
            # Mapear os índices do best tour para os nós reais do grafo
            path_edges = []
            for i in range(len(self.best) - 1):
                # Verificar se os índices estão dentro dos limites
                if self.best[i] < len(node_list) and self.best[i+1] < len(node_list):
                    path_edges.append((node_list[self.best[i]], node_list[self.best[i+1]]))
        
        if path_edges is not None:
            
            # # Desenha apenas as arestas diferentes em vermelho
            nx.draw_networkx_edges(
                self.G, self.pos,
                edgelist=path_edges,
                width=2,
                edge_color=color_tour,
                style='dashed',
                alpha=0.7,
                arrows=True,
                connectionstyle='arc3,rad=0.1'
            )
            
        if not to_video:
            # Adiciona os rótulos das arestas
            edge_labels = {
                (u, v): self.G[u][v].get('weight', 1)
                for u, v in path_edges
            }
            nx.draw_networkx_edge_labels(
                self.G, self.pos,
                edge_labels=edge_labels,
                font_color='black',
                font_size=8,
                label_pos=0.5,
                rotate=True
            )
            
        # Adiciona a legenda
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, linestyle='dashed', label=f'Exploration'),
            Line2D([0], [0], color='white', lw=2, linestyle='dashed', label=f'Best current solution: {self.best_cost:10.1f}')
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
            

        plt.axis('off')
        plt.tight_layout()
        
        if to_video:
            filepath = f"./tmp/{self.name_problem}"
            file_name = f"frame_{self.__class__._get_fn()}.png"
            os.makedirs(filepath, exist_ok=True)
            plt.savefig(f"{filepath}/{file_name}", dpi=300)
            
        elif isinstance(tour, np.ndarray) or tour == 'best':
            filepath = f"./images/{self.name_problem}"
            file_name = f"da_gvns_grafo_best_{uuid.uuid4().hex}.png"
            os.makedirs(filepath, exist_ok=True)
            plt.savefig(f"{filepath}/{file_name}", dpi=300)
                        
        else:
            plt.show()
            
        plt.close()
