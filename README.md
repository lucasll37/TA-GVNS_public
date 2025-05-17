# 🎯 Algoritmos DA-GVNS e TA-GVNS para o Problema do Caixeiro Viajante

## 📋 Visão Geral

Este projeto implementa algoritmos avançados de busca heurística para resolver o Problema do Caixeiro Viajante (TSP) e sua variante assimétrica (ATSP). O foco principal está na implementação de:

1. **DA-GVNS (Double-Adaptive General Variable Neighborhood Search)**: Implementação direta do algoritmo proposto por Karakostas e Sifaleras (2022), que utiliza mecanismos adaptativos em ambas as fases de intensificação e diversificação.

2. **TA-GVNS (Triple-Adaptive General Variable Neighborhood Search)**: Nossa extensão original que adiciona um terceiro mecanismo adaptativo para seleção probabilística de métodos VND.

## 🧩 O Problema do Caixeiro Viajante

O TSP consiste em encontrar a rota mais curta que permite visitar um conjunto de cidades exatamente uma vez e retornar à cidade inicial. Este problema é NP-difícil, tornando abordagens heurísticas como o GVNS particularmente valiosas para instâncias grandes.

## 🏗️ Arquitetura Detalhada do Projeto

```
projeto/
│
├── src/                    # Código-fonte do projeto
│   ├── da_gvns_atsp.py     # Implementação do algoritmo DA-GVNS original
│   ├── ta_gvns_atsp.py     # Implementação do TA-GVNS com seleção adaptativa de VND
│   └── main.py             # Script principal para execução dos experimentos
│
├── data/                   # Dados para testes e referências
│   ├── tsp/                # Instâncias de problemas TSP
│   │   └── brazil58.tsp    # Problema com 58 cidades brasileiras
│   ├── atsp_best_known.txt # Valores ótimos conhecidos para ATSP
│   └── tsp_best_known.txt  # Valores ótimos conhecidos para TSP
│
├── output/                 # Armazenamento dos resultados
│   ├── DA_GVNS/            # Resultados das execuções do DA-GVNS
│   │   └── [problema]/     # Resultados específicos para cada instância
│   │       └── metrics_*.csv # Métricas detalhadas de cada execução
│   └── TA_GVNS/            # Resultados das execuções do TA-GVNS
│       └── [problema]/     # Resultados específicos para cada instância
│           └── metrics_*.csv # Métricas detalhadas de cada execução
│
└── tmp/                    # Arquivos temporários (frames para visualização)
    └── [problema]/         # Frames organizados por problema
        └── frame_*.png     # Imagens sequenciais para criação de vídeos
```

## 💡 Implementação do DA-GVNS (Conforme Proposto por Karakostas e Sifaleras)

### Fundamentos do Algoritmo

O DA-GVNS é uma abordagem metaheurística que aprimora o GVNS tradicional através da introdução de mecanismos adaptativos que reorganizam operadores com base no seu desempenho histórico. Esta abordagem segue os seguintes princípios:

1. **Adaptatividade na Busca Local**: Operadores de busca local são reordenados com base na quantidade de melhorias que produziram
2. **Adaptatividade na Fase de Shake**: Operadores de perturbação são reordenados seguindo o mesmo princípio
3. **Memória de Curto Prazo**: O algoritmo mantém contadores de melhorias para cada operador

### Detalhes da Implementação

#### 1. Inicialização
- O algoritmo carrega uma instância TSP usando a biblioteca TSPLIB95
- Constrói uma matriz de custos e um grafo NetworkX para representação
- Gera uma solução inicial usando a heurística do vizinho mais próximo
- Inicializa contadores para monitorar melhorias de cada operador

```python
def __init__(self, path: str, first_node: Optional[int]=None, benchmark_files: Optional[List[str]]=None) -> None:
    self.name_problem = path.split('/')[-1].split('.')[0]
    self.n, self.cost_matrix, self.G = self._load_tsplib_instance(path)
    self.pos = nx.spring_layout(self.G)
    self.best = self._nearest_neighbor_init(first_node)
    self.best_cost = self._evaluate(self.best)
    self.init_order: List[str] = ['relocate', 'swap', '2opt']
    self.benchmark = self._load_benchmark(self.name_problem, benchmark_files)
```

#### 2. Operadores de Busca Local
O algoritmo implementa três operadores clássicos:

- **Relocate**: Remove um nó de uma posição e o insere em outra
```python
def _local_search_best_move(self, tour: np.ndarray, operator: str):
    # Para o operador 'relocate'
    if operator == 'relocate':
        node = candidate[i]
        candidate = np.delete(candidate, i)
        candidate = np.append(candidate, node)
```

- **Swap**: Troca a posição entre dois nós
```python
# Para o operador 'swap'
elif operator == 'swap':
    candidate[i], candidate[j] = candidate[j], candidate[i]
```

- **2-opt**: Inverte um segmento da rota
```python
# Para o operador '2opt'
elif operator == '2opt':
    candidate[i:j+1] = candidate[i:j+1][::-1]
```

#### 3. Métodos VND (Variable Neighborhood Descent)
O DA-GVNS implementa cinco variantes de VND, cada uma com uma estratégia diferente para percorrer a vizinhança:

- **Sequential VND**: Aplica cada operador sequencialmente
```python
def _sequential_vnd(self, tour: np.ndarray, local_search_order: List[str], local_search_counters: Dict[str,int]):
    improved_any = False
    for op in local_search_order:
        improved = True
        while improved:
            best, delta = self._local_search_best_move(tour, op)
            if delta < 0:
                tour = best
                local_search_counters[op] += 1
                improved_any = True
            else:
                improved = False
    return tour, improved_any
```

- **Pipe VND**: Semelhante ao Sequential, mas reinicia a busca após cada melhoria
- **Nested VND**: Volta ao primeiro operador sempre que encontra uma melhoria
- **Variable VND**: Alterna entre estratégias de melhoria (best, first, random)
- **Cyclic VND**: Continua aplicando operadores em ciclo até não haver melhorias

No entanto, na implementação DA-GVNS conforme o artigo original, apenas o **Pipe VND** é realmente utilizado, enquanto os outros são definidos mas não são selecionados dinamicamente:

```python
# No método search do DA-GVNS
vnd_method_name = "pipe"  # Método fixo no DA-GVNS original
```

#### 4. Mecanismo Adaptativo para Busca Local
O algoritmo implementa um mecanismo para reordenar os operadores de busca local:

```python
def _adapt_order(self, curr_order: List[str], counters: Dict[str,int], no_success: bool) -> List[str]:
    if no_success:
        return self.init_order.copy()
    
    return sorted(curr_order, key=lambda op: counters[op], reverse=True)
```

Este método:
- Retorna à ordem inicial quando não há melhorias
- Reordena os operadores em ordem decrescente de sucesso quando há melhorias

#### 5. Mecanismo Adaptativo para Shake
De forma similar, os operadores de shake são reordenados:

```python
# No método search
shake_ops = self._adapt_order(shake_ops, shake_counters, not shake_any_improv)
```

Esta abordagem permite:
- Dar mais tempo de processamento aos operadores mais eficazes
- Retornar à configuração inicial em caso de estagnação

#### 6. Processo de Busca Completo
O método principal de busca combina todos estes elementos:

```python
def search(self, k_max: int, time_limit: float=60.0, seed: Optional[int]=None, to_print: bool=False, 
        memory_size: int=10):
    # Inicializações
    # ...
    
    while time.time()-start < time_limit:
        # Adaptar ordem dos operadores de shake
        shake_ops = self._adapt_order(shake_ops, shake_counters, not shake_any_improv)
        
        for k in range(k_max):
            op = shake_ops[k%len(shake_ops)]
            s_star = self._shake(self.best, op)
            
            # Adaptar ordem dos operadores de busca local
            local_search_ops = self._adapt_order(local_search_ops, local_search_counters, not local_search_any_improv)
            
            # Aplicar o VND com método pipe fixo
            vnd_method_name = "pipe"  # Fixo no DA-GVNS original
            s_new, improved_ls = self.vnd_methods[vnd_method_name](s_star, local_search_ops, local_search_counters)
            
            # Atualizar melhor solução se necessário
            new_cost = self._evaluate(s_new)
            if new_cost < self.best_cost:
                self.best, self.best_cost = s_new, new_cost
                # ...
```

## 💡 Implementação do TA-GVNS (Nossa Extensão)

O TA-GVNS estende o DA-GVNS adicionando um terceiro mecanismo adaptativo:

### Seleção Adaptativa de Métodos VND

A principal diferença do TA-GVNS está na seleção dinâmica do método VND com base no desempenho histórico:

1. **Memória adaptativa para métodos VND**:
```python
# Memória adaptativa para cada método VND
vnd_memory = {vnd: [0] for vnd in self.vnd_methods.keys()}
```

2. **Cálculo de probabilidades** para cada método VND:
```python
def _calculate_vnd_probabilities(self, vnd_memory: Dict[str, List[float]]) -> Dict[str, float]:
    # Calcular média para cada método
    avg_costs_update = {}
    for vnd, costs in vnd_memory.items():
        avg_costs_update[vnd] = np.abs(sum(costs) / len(costs) if costs else 0.0)
            
    # Calcular probabilidades e garantir valores mínimos
    # ...
    
    return probabilities
```

3. **Seleção probabilística** de métodos VND:
```python
def _select_vnd_method(self, vnd_memory: Dict[str, List[float]]) -> str:
    # Obter as probabilidades calculadas
    probabilities = self._calculate_vnd_probabilities(vnd_memory)
    
    # Selecionar método baseado nas probabilidades
    return np.random.choice(methods, p=probs)
```

4. **Atualização da memória** após cada aplicação:
```python
def _update_vnd_memory(self, vnd_memory: Dict[str, List[float]], vnd_method: str, 
                    cost_update: float, memory_size: int) -> Dict[str, List[float]]:
    vnd_memory[vnd_method].append(cost_update)
    
    # Manter apenas os 'memory_size' custos mais recentes
    if len(vnd_memory[vnd_method]) > memory_size:
        vnd_memory[vnd_method] = vnd_memory[vnd_method][-memory_size:]
                    
    return vnd_memory
```

5. **Aplicação no processo de busca**:
```python
# No método search do TA-GVNS
# Selecionar método VND baseado nas probabilidades
vnd_method_name = self._select_vnd_method(vnd_memory)

# Aplicar o método VND selecionado
s_new, improved_ls = self.vnd_methods[vnd_method_name](s_star, local_search_ops, local_search_counters)

# Atualizar a memória do método VND usado
vnd_memory = self._update_vnd_memory(vnd_memory, vnd_method_name, cost_update, memory_size)
```

## 🚀 Como Executar o Projeto

### Instalação de dependências

Em um ambiente Linux (nativo ou WSL) que possua o gerenciador de ecossistema Anaconda, siga os passos abaixo para instalar as dependências do projeto
(Caso opte por não usar o Linux/Anaconda, interprete o arquivo `Makefile` e faça as ações manualmente):

- Instalação do `Make`:
   ```bash
   sudo apt install make
   ```

- Cria ambiente virtual Python na pasta raiz do projeto (`./.venv`), atualiza o pip e instala as dependências do projeto (`./requirments.txt`):
   ```bash
   make create_env
   ```

- Ativa o ambiente virtual:
   ```bash
   conda activate ./.venv
   ```

### Execução
Para executar o projeto, realize os seguintes passos:

- Edite o arquivo `./src/main.py` com suas preferências:
   ```python
   # -----------------------------------------------------------------------------
   # Configuration
   # -----------------------------------------------------------------------------
    PROBLEM_PATH = "./data/tsp/bayg29.tsp"   # Path to the TSP problem file
    K_MAX = 8                                # Maximum number of iterations for the search
    TIME_LIMIT = 60.0                        # Time limit for the search in seconds
    TO_PRINT = False                         # Whether to print the results
    EPISODES = 5                             # Number of episodes to run for each algorithm
   ```
   
- Execute o comando:
   ```bash
   make run
   ```

### Gerar vídeo
- Caso tenha gerado os frames da busca de caminho (`TO_PRINT = True`), você pode renderizar o vídeo seguindo os seguintes passos:
- Edite o arquivo `./utils/create_video.py` com suas preferências:
   ```python
   # -----------------------------------------------------------------------------
   # Configuration
   # -----------------------------------------------------------------------------
   FRAME_DIR = "./tmp/burma14"  # Directory containing frame images
   SECONDS_PER_FRAME = 0.2      # Duration of each frame in the video  
   ```

- Execute o comando:
   ```bash
   make create_video
   ```

## 📊 Visualização e Análise de Resultados

### Visualização do Grafo e da Solução
O código inclui funcionalidades para visualizar o grafo do problema e as soluções encontradas:

```python
def plot_graph(self, tour: Optional[Union[np.ndarray, str]]=None, color_tour: str = 'red', to_video: bool=False):
    # Desenha o grafo e a rota
    # Salva frames para vídeo se to_video=True
```

### Métricas Coletadas
Os algoritmos coletam métricas detalhadas durante a execução, incluindo:

- **iteration**: Número da iteração
- **time**: Tempo decorrido
- **steps**: Número de passos de busca
- **cost**: Custo da solução atual
- **best_cost**: Melhor custo encontrado
- **benchmark**: Valor ótimo conhecido
- **vnd_method**: Método VND utilizado
- **shake_operator**: Operador de perturbação usado
- **improvement**: Melhoria de custo obtida
- **prob_[método]**: Probabilidades de seleção dos métodos VND (no TA-GVNS)

Estas métricas são salvas em arquivos CSV para posterior análise.

## 🔍 Como Interpretar os Resultados

### Estrutura dos Arquivos de Saída
Para cada execução, um arquivo CSV é gerado contendo todas as métricas:

```
output/
├── DA_GVNS/
│   └── brazil58/
│       ├── metrics_0.csv
│       ├── metrics_1.csv
│       └── ...
└── TA_GVNS/
    └── brazil58/
        ├── metrics_0.csv
        ├── metrics_1.csv
        └── ...
```

### Análise Comparativa
Para comparar os algoritmos, você pode:

1. **Analisar o custo final**: Comparar o melhor custo encontrado por cada algoritmo
2. **Examinar a velocidade de convergência**: Observar como o custo diminui ao longo do tempo
3. **Verificar a utilização de métodos VND**: No TA-GVNS, analisar quais métodos VND foram mais frequentemente selecionados
4. **Comparar a adaptatividade**: Observar como as probabilidades de seleção mudam ao longo da execução

## 📚 Base Teórica

### O Artigo Original
Este projeto implementa e estende o algoritmo DA-GVNS proposto por Karakostas e Sifaleras (2022) em seu artigo "A Double-Adaptive General Variable Neighborhood Search algorithm for the solution of the Traveling Salesman Problem". As principais contribuições do artigo são:

1. A integração de mecanismos adaptativos em ambas as fases do GVNS
2. A demonstração de que esta abordagem supera variantes convencionais e de adaptação única
3. A validação em benchmarks de instâncias simétricas e assimétricas do TSP

### A Extensão TA-GVNS
Nossa extensão TA-GVNS adiciona um terceiro nível de adaptação, permitindo:

1. Seleção dinâmica entre diferentes variantes de VND
2. Ajuste automático do método de busca local mais adequado para cada fase da execução
3. Balanceamento entre exploração e explotação através de mínimos probabilísticos

---

Este projeto demonstra como técnicas avançadas de otimização combinatória podem ser implementadas e estendidas para resolver problemas complexos. O TA-GVNS representa uma evolução natural do DA-GVNS, trazendo maior adaptabilidade e potencialmente melhores resultados para instâncias variadas do Problema do Caixeiro Viajante.
