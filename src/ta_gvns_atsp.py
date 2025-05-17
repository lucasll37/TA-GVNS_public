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

class TA_GVNS:
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
        self.vnd_methods = {
            'sequential': self._sequential_vnd,
            'pipe': self._pipe_vnd,
            'nested': self._nested_vnd,
            'variable': self._variable_vnd,
            'cyclic': self._cyclic_vnd
        }
        
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
                
                # if self.to_print:
                #     self.plot_graph(tour=candidate, color_tour='red', to_video=True)
                
                if delta < best_delta:
                    best_delta = delta
                    best = candidate.copy()
                    
                if operator == 'relocate':
                    break
                    
        return best, best_delta
    
    def _local_search_first_move(self, tour: np.ndarray, operator: str) -> Tuple[np.ndarray, float]:
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
                
                # if self.to_print:
                #     self.plot_graph(tour=candidate, color_tour='red', to_video=True)
                
                if delta < best_delta:
                    best_delta = delta
                    best = candidate.copy()
                    
                    return best, best_delta
                    
                if operator == 'relocate':
                    break
                    
        return best, best_delta
    
    def _local_search_random_move(self, tour: np.ndarray, operator: str) -> Tuple[np.ndarray, float]:
        cost0 = self._evaluate(tour)
        
        i, j = sorted(random.sample(range(0, self.n), 2))
                    
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
            
        tour = np.append(candidate, candidate[0])
        delta = self._evaluate(candidate) - cost0
        
        # if self.to_print:
        #     self.plot_graph(tour=tour, color_tour='red', to_video=True)
        
        return tour, delta        
    
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
    
    def _pipe_vnd(self, tour: np.ndarray, local_search_order: List[str], local_search_counters: Dict[str,int]) -> Tuple[np.ndarray, bool]:
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
    
    def _nested_vnd(self, tour: np.ndarray, local_search_order: List[str], local_search_counters: Dict[str,int]) -> Tuple[np.ndarray, bool]:
        improved_any = False
        k = 0
        
        while k < len(local_search_order):
            improved = True
            while improved:
                best, delta = self._local_search_best_move(tour, local_search_order[k])
                if delta < 0:
                    tour = best
                    local_search_counters[local_search_order[k]] += 1
                    improved_any = True
                    
                    if self.to_print:
                        self.plot_graph(tour=tour, color_tour='red', to_video=True)
                    
                else:
                    improved = False
            
            if improved:
                k = 0  # Se melhorou, volta ao início
            else:
                k += 1  # Se não melhorou, avança
                
        return tour, improved_any
    
    
    def _variable_vnd(self, tour: np.ndarray, local_search_order: List[str], local_search_counters: Dict[str,int]) -> Tuple[np.ndarray, bool]:
        improved_any = False
        
        for op in local_search_order:
            improved = True
            while improved:
                strategy = random.choice(['best', 'first', 'random'])
                
                if strategy == 'best':
                    best, delta = self._local_search_best_move(tour, op)
                    
                elif strategy == 'first':
                    best, delta = self._local_search_first_move(tour, op)
                    
                else:  # random
                    best, delta = self._local_search_random_move(tour, op)
                    
                if delta < 0:
                    tour = best
                    local_search_counters[op] += 1
                    improved_any = True
                    
                    if self.to_print:
                        self.plot_graph(tour=tour, color_tour='red', to_video=True)
                    
                else:
                    improved = False
                    
        return tour, improved_any
    
    def _cyclic_vnd(self, tour: np.ndarray, local_search_order: List[str], local_search_counters: Dict[str,int]) -> Tuple[np.ndarray, bool]:
        improved_any = False
        improved = True
        
        while improved:
            improved = False
            for op in local_search_order:
                local_improved = True
                
                while local_improved:
                    best, delta = self._local_search_best_move(tour, op)
                    if delta < 0:
                        improved = True
                        tour = best
                        local_search_counters[op] += 1
                        improved_any = True
                        
                        if self.to_print:
                            self.plot_graph(tour=tour, color_tour='red', to_video=True)
                            
                    else:
                        local_improved = False
                        
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
            'vnd_method': [],
            'shake_operator': [],
            'improvement': []
        }
        
        # Adicionar colunas para as probabilidades de cada método VND
        for vnd_method in self.vnd_methods.keys():
            metrics_data[f'prob_{vnd_method}'] = []
        
        iteration_count = 0
        shake_ops = self.init_order.copy()
        local_search_ops = self.init_order.copy()
        
        shake_counters = {op:0 for op in shake_ops}
        local_search_counters = {op:0 for op in local_search_ops}
        
        # Memória adaptativa para cada método VND
        vnd_memory = {vnd: [0] for vnd in self.vnd_methods.keys()}
        
        shake_any_improv = False
        local_search_any_improv = False
        
        # Calcular probabilidades dos métodos VND
        vnd_probs = self._calculate_vnd_probabilities(vnd_memory)
        # Registro básico inicial
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
        
        # Adicionar probabilidades de cada método VND
        for vnd, prob in vnd_probs.items():
            metrics_entry[f'prob_{vnd}'] = prob
        
        # Adicionar entrada ao dataframe
        for key, value in metrics_entry.items():
            metrics_data[key].append(value)
        
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
                
                # Calcular probabilidades dos métodos VND
                vnd_probs = self._calculate_vnd_probabilities(vnd_memory)
                
                # Selecionar método VND baseado nas probabilidades
                # vnd_method_name = "pipe"
                vnd_method_name = self._select_vnd_method(vnd_memory)
                
                # Aplicar o método VND selecionado
                old_step = self.steps
                s_new, improved_ls = self.vnd_methods[vnd_method_name](s_star, local_search_ops, local_search_counters)
                
                if improved_ls:
                    local_search_any_improv = True
                    
                # Avaliar a nova solução
                new_cost = self._evaluate(s_new)
                cost_update = new_cost - self.best_cost
                cost_update_weighted = cost_update / (self.steps - old_step)
                
                # Atualizar a memória do método VND usado
                vnd_memory = self._update_vnd_memory(vnd_memory, vnd_method_name, cost_update_weighted, memory_size)
                
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
                    'vnd_method': vnd_method_name,
                    'shake_operator': op,
                    'improvement': cost_update
                }
                
                # Adicionar probabilidades de cada método VND
                for vnd, prob in vnd_probs.items():
                    metrics_entry[f'prob_{vnd}'] = prob
                
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
    
    def _calculate_vnd_probabilities(self, vnd_memory: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calcula as probabilidades de seleção para cada método VND
        
        Args:
            vnd_memory: Dicionário com custos recentes para cada método VND
            
        Returns:
            Dicionário com as probabilidades de cada método
        """
        # Calcular média para cada método
        avg_costs_update = {}
        for vnd, costs in vnd_memory.items():
            avg_costs_update[vnd] = sum(costs) / len(costs) if costs else 0.0
            
        delta = min(avg_costs_update.values())
        
        for vnd, costs in vnd_memory.items():
            avg_costs_update[vnd] -= delta
                
        # Calcular soma dos custos médios
        sum_costs_update = sum(avg_costs_update.values()) + 1e-6  # Para evitar divisão por zero
        
        # Calcular probabilidades iniciais
        methods = list(vnd_memory.keys())
        probs = [cost_update/sum_costs_update for cost_update in avg_costs_update.values()]
        
        # Ajustar probabilidades para garantir valores mínimos
        temp = 0
        richs = []
        for i in range(len(probs)):
            if probs[i] < 0.1:
                temp += 0.1 - probs[i]
                probs[i] = 0.1
                    
            elif probs[i] > 0.25:
                richs.append(i)
                    
        for rich in richs:
            probs[rich] -= temp / len(richs) if richs else 0

        # Normalizar para que a soma seja 1
        sum_probs = sum(probs)       
        probs = [p/sum_probs for p in probs]
        
        # Criar dicionário de probabilidades
        probabilities = {methods[i]: probs[i] for i in range(len(methods))}
        
        return probabilities
    
    def _select_vnd_method(self, vnd_memory: Dict[str, List[float]]) -> str:
        # Obter as probabilidades calculadas
        probabilities = self._calculate_vnd_probabilities(vnd_memory)
        
        # Preparar para seleção ponderada
        methods = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Selecionar método baseado nas probabilidades
        return np.random.choice(methods, p=probs)

    def _update_vnd_memory(self, vnd_memory: Dict[str, List[float]], vnd_method: str, 
                        cost_update: float, memory_size: int) -> Dict[str, List[float]]:

        vnd_memory[vnd_method].append(cost_update)
        
        # Manter apenas os 'memory_size' custos mais recentes
        if len(vnd_memory[vnd_method]) > memory_size:
            vnd_memory[vnd_method] = vnd_memory[vnd_method][-memory_size:]
            
        return vnd_memory
                        
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
            # best_edges = [(self.best[i] + 1, self.best[i+1] + 1) for i in range(len(self.best) - 1)]
            # nx.draw_networkx_edges(
            #     self.G, self.pos,
            #     edgelist=best_edges,
            #     width=2,
            #     edge_color='black',
            #     style='dashed',
            #     alpha=1,
            #     arrows=True,
            #     connectionstyle='arc3,rad=0.1'
            # )
            
            
            # different_edges = [edge for edge in path_edges if edge not in best_edges]
        
            # # Desenha apenas as arestas diferentes em vermelho
            # if different_edges:
            #     nx.draw_networkx_edges(
            #         self.G, self.pos,
            #         edgelist=different_edges,
            #         width=2,
            #         edge_color='red',
            #         style='dashed',
            #         alpha=0.7,
            #         arrows=True,
            #         connectionstyle='arc3,rad=0.1'
            #     )
            
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
            file_name = f"ta_gvns_grafo_best_{uuid.uuid4().hex}.png"
            os.makedirs(filepath, exist_ok=True)
            plt.savefig(f"{filepath}/{file_name}", dpi=300)
                        
        else:
            plt.show()
            
        plt.close()
