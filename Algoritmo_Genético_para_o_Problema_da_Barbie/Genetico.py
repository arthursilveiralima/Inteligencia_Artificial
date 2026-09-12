import random

class AlgoritmoGeneticoTSP:
    """
    Implementa um Algoritmo Genético para resolver o Problema do Caixeiro Viajante (TSP).
    """
    def __init__(self, matriz_custos, num_amigos, tam_pop, geracoes, tx_cruz, tx_mut, elitismo):
        self.matriz = matriz_custos
        self.num_amigos = num_amigos
        self.tam_pop = tam_pop
        self.geracoes = geracoes
        self.tx_cruz = tx_cruz
        self.tx_mut = tx_mut
        self.elitismo = elitismo

        self.populacao = []
        self.melhor_rota = None
        self.menor_custo = float("inf")
        self.historico_custos = []
        self.geracao_estabilizacao = 0

    def inicializar_populacao(self):
        """Gera a população inicial com permutações aleatórias dos amigos."""
        base_amigos = list(range(self.num_amigos))
        self.populacao = []
        for _ in range(self.tam_pop):
            individuo = base_amigos[:]
            random.shuffle(individuo)
            self.populacao.append(individuo)

    def avaliar_custo_rota(self, individuo):
        """Calcula o custo total: Casa -> Amigos -> Casa."""
        custo_total = 0
        ponto_atual = 0  # 0 representa a Casa da Barbie na matriz de distâncias
        
        for amigo in individuo:
            prox_ponto = amigo + 1  # +1 pois os amigos começam no índice 1 da matriz
            custo_total += self.matriz[ponto_atual][prox_ponto]
            ponto_atual = prox_ponto
            
        # Adiciona o custo de retorno para casa
        custo_total += self.matriz[ponto_atual][0]
        return custo_total

    def calcular_aptidao(self, individuo):
        """Fitness é o inverso do custo (minimizar custo = maximizar fitness)."""
        return 1.0 / (1.0 + self.avaliar_custo_rota(individuo))

    def selecao_por_roleta(self):
        """Seleciona indivíduos com probabilidade proporcional ao fitness."""
        aptidoes = [self.calcular_aptidao(ind) for ind in self.populacao]
        soma_aptidao = sum(aptidoes)
        
        selecionados = []
        for _ in range(self.tam_pop):
            ponteiro = random.uniform(0, soma_aptidao)
            acumulador = 0
            for ind, apt in zip(self.populacao, aptidoes):
                acumulador += apt
                if acumulador >= ponteiro:
                    selecionados.append(ind[:])
                    break
        return selecionados

    def cruzamento_ox(self, pai1, pai2):
        """Realiza o Order Crossover (OX) para preservar valores de permutações."""
        tamanho = len(pai1)
        ponto_corte1, ponto_corte2 = sorted(random.sample(range(tamanho), 2))
        
        filho = [None] * tamanho
        
        # Herda o segmento contíguo do pai1
        filho[ponto_corte1:ponto_corte2 + 1] = pai1[ponto_corte1:ponto_corte2 + 1]
        
        # Preenche o restante com a ordem exata do pai2
        genes_faltantes = [g for g in pai2 if g not in filho]
        
        idx_faltante = 0
        for i in range(tamanho):
            if filho[i] is None:
                filho[i] = genes_faltantes[idx_faltante]
                idx_faltante += 1
                
        return filho

    def mutacao_swap(self, individuo):
        """Troca a posição de dois genes aleatoriamente."""
        if random.random() < self.tx_mut:
            i, j = random.sample(range(len(individuo)), 2)
            individuo[i], individuo[j] = individuo[j], individuo[i]
        return individuo

    def salvar_melhor_individuo(self):
        """Atualiza o recorde global da melhor rota encontrada na população."""
        for ind in self.populacao:
            custo = self.avaliar_custo_rota(ind)
            if custo < self.menor_custo:
                self.menor_custo = custo
                self.melhor_rota = ind[:]

    def otimizar(self):
        """Executa o loop de gerações do Algoritmo Genético."""
        self.inicializar_populacao()
        self.salvar_melhor_individuo()
        self.historico_custos.append(self.menor_custo)

        for _ in range(self.geracoes):
            # Preserva a elite
            populacao_ordenada = sorted(self.populacao, key=self.avaliar_custo_rota)
            elite = [ind[:] for ind in populacao_ordenada[:self.elitismo]]

            pais_selecionados = self.selecao_por_roleta()
            nova_geracao = []

            # Cruzamento
            for i in range(0, self.tam_pop, 2):
                pai_a = pais_selecionados[i]
                pai_b = pais_selecionados[i + 1] if (i + 1) < self.tam_pop else pais_selecionados[0]

                if random.random() < self.tx_cruz:
                    filho1 = self.cruzamento_ox(pai_a, pai_b)
                    filho2 = self.cruzamento_ox(pai_b, pai_a)
                else:
                    filho1 = pai_a[:]
                    filho2 = pai_b[:]

                # Mutação
                nova_geracao.append(self.mutacao_swap(filho1))
                if len(nova_geracao) < self.tam_pop:
                    nova_geracao.append(self.mutacao_swap(filho2))

            # Restaura a elite na nova geração
            for i in range(self.elitismo):
                nova_geracao[i] = elite[i]

            self.populacao = nova_geracao
            self.salvar_melhor_individuo()
            self.historico_custos.append(self.menor_custo)

        self.geracao_estabilizacao = self.historico_custos.index(self.menor_custo)
        return self.melhor_rota
