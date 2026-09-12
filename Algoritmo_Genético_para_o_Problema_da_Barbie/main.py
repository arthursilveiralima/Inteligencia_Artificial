import random
import time
from itertools import permutations
import matplotlib.pyplot as plt

from Mapa import carregar_matriz_mapa, eh_posicao_valida, CASA_BARBIE
from Busca import executar_a_estrela
from Genetico import AlgoritmoGeneticoTSP

# =============================================================================
# CONFIGURAÇÕES GERAIS E HIPERPARÂMETROS
# =============================================================================

# Matricula
MATRICULA_ALUNO = 2024105231940008 

TAMANHO_POP = 100
MAX_GERACOES = 300
TAXA_CRUZAMENTO = 0.85
TAXA_MUTACAO = 0.04
ELITISMO = 2

# =============================================================================
# PREPARAÇÃO DO PROBLEMA
# =============================================================================

def sortear_amigos_barbie(mapa, qtd_amigos, semente):
    """Sorteia N posições aleatórias válidas no mapa usando a matrícula como seed."""
    random.seed(semente)
    posicoes_livres = []
    
    for l in range(mapa.shape[0]):
        for c in range(mapa.shape[1]):
            # Verifica se é chão caminhável e se não é a própria casa da Barbie
            if eh_posicao_valida(mapa, l, c) and (l, c) != CASA_BARBIE:
                posicoes_livres.append((l, c))

    escolhidos = random.sample(posicoes_livres, qtd_amigos)
    return {idx: pos for idx, pos in enumerate(escolhidos)}

def construir_matriz_distancias(mapa, dicionario_amigos):
    """Roda o A* entre a Casa e os Amigos para criar a matriz de adjacência do TSP."""
    nos = [CASA_BARBIE] + list(dicionario_amigos.values())
    qtd_nos = len(nos)
    
    matriz = [[0] * qtd_nos for _ in range(qtd_nos)]

    for i in range(qtd_nos):
        for j in range(qtd_nos):
            if i != j:
                # Chama o A* importado do Busca.py
                _, custo_viagem = executar_a_estrela(mapa, nos[i], nos[j])
                matriz[i][j] = custo_viagem

    return matriz

def solver_forca_bruta(matriz, num_amigos):
    """Testa todas as permutações possíveis (Cuidado: fatorial de N)."""
    melhor_ordem = None
    menor_custo = float("inf")

    for permutacao in permutations(range(num_amigos)):
        rota = [0] + [a + 1 for a in permutacao] + [0]
        custo_atual = 0
        
        for i in range(len(rota) - 1):
            custo_atual += matriz[rota[i]][rota[i + 1]]
            
        if custo_atual < menor_custo:
            menor_custo = custo_atual
            melhor_ordem = permutacao

    return melhor_ordem, menor_custo

def plotar_grafico(historico, titulo):
    """Salva o gráfico de convergência do algoritmo como uma imagem."""
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(historico)), historico, color='purple', linewidth=2)
    plt.xlabel("Gerações")
    plt.ylabel("Custo da Melhor Rota")
    plt.title(titulo)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    nome_arquivo = f"{titulo.replace(' ', '_').lower()}.png"
    plt.savefig(nome_arquivo)
    print(f"    [Gráfico salvo como: {nome_arquivo}]")
    plt.close()

# =============================================================================
# EXPERIMENTOS
# =============================================================================

def rodar_experimento_a(mapa):
    print("\n" + "="*50)
    print(" EXPERIMENTO A (N = 8 Amigos)")
    print("="*50)

    n_amigos = 8
    amigos_pos = sortear_amigos_barbie(mapa, n_amigos, MATRICULA_ALUNO)
    
    print("\n[+] Posições sorteadas:")
    for amigo, pos in amigos_pos.items():
        print(f"    Amigo {amigo}: {pos}")

    print("\n[+] Executando A* para montar matriz de distâncias...")
    matriz = construir_matriz_distancias(mapa, amigos_pos)

    print("\n[+] Iniciando o Algoritmo Genético...")
    ag = AlgoritmoGeneticoTSP(matriz, n_amigos, TAMANHO_POP, MAX_GERACOES, TAXA_CRUZAMENTO, TAXA_MUTACAO, ELITISMO)
    t_inicio = time.perf_counter()
    ag.otimizar()
    t_ag = time.perf_counter() - t_inicio

    print(f"    Melhor rota encontrada: {ag.melhor_rota}")
    print(f"    Custo encontrado: {ag.menor_custo}")
    print(f"    Tempo de execução: {t_ag:.4f} segundos")

    print("\n[+] Iniciando a Força Bruta (testando todas as permutações)...")
    t_inicio = time.perf_counter()
    rota_fb, custo_fb = solver_forca_bruta(matriz, n_amigos)
    t_fb = time.perf_counter() - t_inicio

    print(f"    Melhor rota real (FB): {rota_fb}")
    print(f"    Custo ótimo: {custo_fb}")
    print(f"    Tempo de execução: {t_fb:.4f} segundos")

    print("\n[+] Conclusão:")
    if ag.menor_custo == custo_fb:
        print("    -> SUCESSO: O Algoritmo Genético encontrou a solução ótima!")
    else:
        erro = ((ag.menor_custo - custo_fb) / custo_fb) * 100
        print(f"    -> O AG ficou {erro:.2f}% acima do custo ideal.")
        
    return ag

def rodar_experimento_b(mapa):
    print("\n" + "="*50)
    print(" EXPERIMENTO B (N = 15 Amigos)")
    print("="*50)

    n_amigos = 15
    amigos_pos = sortear_amigos_barbie(mapa, n_amigos, MATRICULA_ALUNO)

    print("\n[+] Executando A* para montar matriz de distâncias (pode levar alguns segundos)...")
    matriz = construir_matriz_distancias(mapa, amigos_pos)

    print("\n[+] Iniciando o Algoritmo Genético (15! possibilidades)...")
    ag = AlgoritmoGeneticoTSP(matriz, n_amigos, TAMANHO_POP, MAX_GERACOES, TAXA_CRUZAMENTO, TAXA_MUTACAO, ELITISMO)
    t_inicio = time.perf_counter()
    ag.otimizar()
    t_ag = time.perf_counter() - t_inicio

    print(f"    Melhor rota encontrada: {ag.melhor_rota}")
    print(f"    Melhor custo: {ag.menor_custo}")
    print(f"    Geração de convergência: {ag.geracao_estabilizacao}")
    print(f"    Tempo de execução: {t_ag:.4f} segundos")

    return ag

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    # Carrega o mapa do arquivo Mapa.py
    mapa_numpy = carregar_matriz_mapa()
    
    # Roda o primeiro experimento e salva o gráfico
    ag_exp_a = rodar_experimento_a(mapa_numpy)
    plotar_grafico(ag_exp_a.historico_custos, "Convergência_AG_-_Experimento_A")
    
    # Roda o segundo experimento e salva o gráfico
    ag_exp_b = rodar_experimento_b(mapa_numpy)
    plotar_grafico(ag_exp_b.historico_custos, "Convergência_AG_-_Experimento_B")

if __name__ == "__main__":
    main()
