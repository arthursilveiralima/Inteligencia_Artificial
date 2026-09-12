import heapq
from Mapa import eh_posicao_valida

def heuristica_manhattan(a, b):
    """Calcula a distância de Manhattan (movimentos ortogonais na grade)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def executar_a_estrela(mapa, inicio, fim):
    """
    Encontra o caminho mais barato entre dois pontos usando A*.
    Retorna a rota (lista de coordenadas) e o custo total.
    """
    fronteira = []
    heapq.heappush(fronteira, (0, inicio))
    
    custo_acumulado = {inicio: 0}
    came_from = {inicio: None}
    
    # Movimentos: Direita, Baixo, Esquerda, Cima
    direcoes = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while fronteira:
        _, atual = heapq.heappop(fronteira)
        
        if atual == fim:
            break
            
        for dx, dy in direcoes:
            l, c = atual[0] + dx, atual[1] + dy
            vizinho = (l, c)
            
            if eh_posicao_valida(mapa, l, c):
                # O custo numérico já foi processado e armazenado na matriz numpy em mapa.py
                custo_terreno = mapa[l][c] 
                novo_custo = custo_acumulado[atual] + custo_terreno
                
                if vizinho not in custo_acumulado or novo_custo < custo_acumulado[vizinho]:
                    custo_acumulado[vizinho] = novo_custo
                    prioridade = novo_custo + heuristica_manhattan(fim, vizinho)
                    heapq.heappush(fronteira, (prioridade, vizinho))
                    came_from[vizinho] = atual
                    
    caminho = []
    if fim in came_from:
        atual = fim
        while atual is not None:
            caminho.append(atual)
            atual = came_from[atual]
        caminho.reverse()
        
    return caminho, custo_acumulado.get(fim, float('inf'))
