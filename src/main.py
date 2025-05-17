import os
from da_gvns_atsp import DA_GVNS
from ta_gvns_atsp import TA_GVNS


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROBLEM_PATH = "./data/tsp/bayg29.tsp"   # Path to the TSP problem file
K_MAX = 8                                # Maximum number of iterations for the search
TIME_LIMIT = 6.0                       # Time limit for the search in seconds
TO_PRINT = False                         # Whether to print the results
EPISODES = 1                            # Number of episodes to run for each algorithm
# -----------------------------------------------------------------------------


problem_name = PROBLEM_PATH.split("/")[-1].split(".")[0]


print(f"Executando Tripe Adaptive GVNS ...")
ta_gvns = TA_GVNS(PROBLEM_PATH, benchmark_files=['./data/atsp_best_known.txt', './data/tsp_best_known.txt'])
history = ta_gvns.search(k_max=K_MAX, time_limit=TIME_LIMIT, to_print=TO_PRINT)
ta_gvns.plot_graph('best')

dir = f'./output/{ta_gvns.__class__.__name__}/{problem_name}'
os.makedirs(dir, exist_ok=True)
history['metrics_df'].to_csv(f'{dir}/metrics.csv', index=False)

print(f"Tempo de busca: {history['search_time']:.2f} segundos")
print(f"Melhor tour encontrado: {history['best']}")
print(f"Custo do melhor tour encontrado: {history['best_cost']} apos {history['steps']} passos")
    
# -----------------------------------------------------------------------------
print(f"Executando Double Adaptive GVNS ...") 
da_gvns = DA_GVNS(PROBLEM_PATH, benchmark_files=['./data/atsp_best_known.txt', './data/tsp_best_known.txt'])
history = da_gvns.search(k_max=K_MAX, time_limit=TIME_LIMIT, to_print=TO_PRINT)
da_gvns.plot_graph('best')

dir = f'./output/{da_gvns.__class__.__name__}/{problem_name}'
os.makedirs(dir, exist_ok=True)
history['metrics_df'].to_csv(f'{dir}/metrics.csv', index=False)

print(f"Tempo de busca: {history['search_time']:.2f} segundos")
print(f"Melhor tour encontrado: {history['best']}")
print(f"Custo do melhor tour encontrado: {history['best_cost']} apos {history['steps']} passos")
