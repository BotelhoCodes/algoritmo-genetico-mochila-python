import random
import time
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

# Usamos namedtuple para organizar melhor os dados de cada item
Item = namedtuple("Item", ["nome", "peso", "valor"])

# --- CONJUNTO DE ITENS ORIGINAL ---
itens_originais = [
    Item("Item 1", 5, 10), Item("Item 2", 8, 12), Item("Item 3", 3, 7),
    Item("Item 4", 2, 5), Item("Item 5", 7, 15), Item("Item 6", 4, 8),
    Item("Item 7", 9, 20), Item("Item 8", 1, 3), Item("Item 9", 6, 11),
    Item("Item 10", 10, 22), Item("Item 11", 5, 9), Item("Item 12", 8, 14),
    Item("Item 13", 3, 6), Item("Item 14", 2, 4), Item("Item 15", 7, 16),
    Item("Item 16", 4, 9), Item("Item 17", 9, 18), Item("Item 18", 1, 2),
    Item("Item 19", 6, 13), Item("Item 20", 10, 25)
]

# --- CONJUNTO DE ITENS MODIFICADO (5 itens alterados) ---
itens_modificados = [
    Item("Item 1", 5, 10), Item("Item 2", 8, 12), Item("Item 3", 3, 7),
    Item("Item 4", 2, 5), Item("Item 5", 7, 15), Item("Item 6", 4, 8),
    Item("Item 7", 9, 20), Item("Item 8", 1, 3), Item("Item 9", 6, 11),
    Item("Item 10", 10, 22), Item("Item 11", 2, 15),  # Modificado: melhor valor/peso
    Item("Item 12", 5, 20),  # Modificado: melhor valor/peso
    Item("Item 13", 3, 6), Item("Item 14", 2, 4),
    Item("Item 15", 4, 18),  # Modificado: melhor valor/peso
    Item("Item 16", 4, 9), Item("Item 17", 9, 18),
    Item("Item 18", 1, 8),   # Modificado: melhor valor/peso
    Item("Item 19", 6, 13),
    Item("Item 20", 8, 30)   # Modificado: melhor valor/peso
]

# --- PARÂMETROS DO ALGORITMO GENÉTICO ---
TAMANHO_POPULACAO = 100
NUM_GERACOES = 200
TAXA_CROSSOVER = 0.85
TAXA_MUTACAO = 0.01
TAMANHO_TORNEIO = 3

class AlgoritmoGeneticoMochila:
    def __init__(self, itens, capacidade_mochila):
        self.itens = itens
        self.capacidade_mochila = capacidade_mochila
        self.num_itens = len(itens)
        self.melhor_fitness_historico = []
        self.media_fitness_historico = []

    def criar_individuo(self):
        """Cria um indivíduo (cromossomo) aleatório."""
        return [random.randint(0, 1) for _ in range(self.num_itens)]

    def criar_populacao_inicial(self):
        """Cria a população inicial."""
        return [self.criar_individuo() for _ in range(TAMANHO_POPULACAO)]

    def calcular_fitness(self, individuo):
        """Calcula o fitness (aptidão) de um indivíduo."""
        peso_total = 0
        valor_total = 0

        for i, gene in enumerate(individuo):
            if gene == 1:
                peso_total += self.itens[i].peso
                valor_total += self.itens[i].valor

        if peso_total > self.capacidade_mochila:
            return 0
        else:
            return valor_total

    def selecao_por_torneio(self, populacao, fitness_pop):
        """Seleciona um indivíduo usando Seleção por Torneio."""
        competidores_indices = random.sample(range(len(populacao)), TAMANHO_TORNEIO)
        indice_vencedor = competidores_indices[0]
        melhor_fitness = fitness_pop[indice_vencedor]

        for i in competidores_indices[1:]:
            if fitness_pop[i] > melhor_fitness:
                melhor_fitness = fitness_pop[i]
                indice_vencedor = i

        return populacao[indice_vencedor]

    def crossover_ponto_unico(self, pai1, pai2):
        """Realiza o Crossover de Ponto Único."""
        if random.random() > TAXA_CROSSOVER:
            return pai1[:], pai2[:]

        ponto_corte = random.randint(1, self.num_itens - 1)
        filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
        filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]

        return filho1, filho2

    def mutacao(self, individuo):
        """Aplica mutação no indivíduo."""
        for i in range(len(individuo)):
            if random.random() < TAXA_MUTACAO:
                individuo[i] = 1 - individuo[i]
        return individuo

    def executar(self):
        """Executa o ciclo completo do Algoritmo Genético."""
        populacao = self.criar_populacao_inicial()
        melhor_solucao_global = None
        melhor_fitness_global = -1

        for geracao in range(NUM_GERACOES):
            fitness_populacao = [self.calcular_fitness(ind) for ind in populacao]

            # Estatísticas
            melhor_fitness_geracao = max(fitness_populacao)
            media_fitness_geracao = sum(fitness_populacao) / len(fitness_populacao)
            self.melhor_fitness_historico.append(melhor_fitness_geracao)
            self.media_fitness_historico.append(media_fitness_geracao)

            # Atualizar melhor global
            if melhor_fitness_geracao > melhor_fitness_global:
                melhor_fitness_global = melhor_fitness_geracao
                melhor_solucao_global = populacao[fitness_populacao.index(melhor_fitness_geracao)]

            # Criar nova geração
            nova_populacao = []

            # Elitismo
            nova_populacao.append(populacao[fitness_populacao.index(melhor_fitness_geracao)][:])

            while len(nova_populacao) < TAMANHO_POPULACAO:
                pai1 = self.selecao_por_torneio(populacao, fitness_populacao)
                pai2 = self.selecao_por_torneio(populacao, fitness_populacao)

                filho1, filho2 = self.crossover_ponto_unico(pai1, pai2)

                filho1 = self.mutacao(filho1)
                filho2 = self.mutacao(filho2)

                nova_populacao.append(filho1)
                if len(nova_populacao) < TAMANHO_POPULACAO:
                    nova_populacao.append(filho2)

            populacao = nova_populacao

        return melhor_solucao_global, melhor_fitness_global

    def plotar_evolucao(self):
        """Plota o gráfico da evolução do fitness."""
        plt.figure(figsize=(12, 6))
        geracoes = range(1, len(self.melhor_fitness_historico) + 1)

        plt.plot(geracoes, self.melhor_fitness_historico,
                label='Melhor Fitness', linewidth=2, color='blue')
        plt.plot(geracoes, self.media_fitness_historico,
                label='Média Fitness', linewidth=2, color='red', linestyle='--')

        plt.xlabel('Geração')
        plt.ylabel('Fitness (Valor)')
        plt.title('Evolução do Fitness ao Longo das Gerações')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def executar_experimentos_multiplos(itens, capacidade, num_execucoes=10):
    """Executa múltiplas execuções para análise estatística."""
    tempos = []
    melhores_valores = []

    for i in range(num_execucoes):
        inicio = time.time()

        ag = AlgoritmoGeneticoMochila(itens, capacidade)
        melhor_solucao, melhor_valor = ag.executar()

        fim = time.time()
        tempo_execucao = fim - inicio

        tempos.append(tempo_execucao)
        melhores_valores.append(melhor_valor)

        # Plotar apenas na última execução
        if i == num_execucoes - 1:
            ag.plotar_evolucao()

    return np.mean(tempos), np.mean(melhores_valores), np.std(melhores_valores)

def analisar_capacidades_diferentes():
    """Análise para diferentes capacidades da mochila."""
    print("=== ANÁLISE: VARIAÇÃO DA CAPACIDADE DA MOCHILA ===\n")

    capacidades = [30, 40, 50, 60, 70]
    resultados = []

    for capacidade in capacidades:
        print(f"Executando para capacidade: {capacidade}kg")
        tempo_medio, valor_medio, desvio_valor = executar_experimentos_multiplos(
            itens_originais, capacidade, 5
        )

        resultados.append({
            'capacidade': capacidade,
            'tempo_medio': tempo_medio,
            'valor_medio': valor_medio,
            'desvio_valor': desvio_valor
        })

        print(f"  Valor médio: {valor_medio:.2f} (±{desvio_valor:.2f})")
        print(f"  Tempo médio: {tempo_medio:.2f}s\n")

    return resultados

def analisar_conjuntos_diferentes():
    """Análise para diferentes conjuntos de itens."""
    print("=== ANÁLISE: DIFERENTES CONJUNTOS DE ITENS ===\n")

    capacidade_fixa = 50
    conjuntos = [
        ("Original", itens_originais),
        ("Modificado", itens_modificados)
    ]

    resultados = []

    for nome, itens in conjuntos:
        print(f"Executando para conjunto: {nome}")
        tempo_medio, valor_medio, desvio_valor = executar_experimentos_multiplos(
            itens, capacidade_fixa, 5
        )

        # Calcular densidade média (valor/peso)
        densidades = [item.valor / item.peso for item in itens]
        densidade_media = sum(densidades) / len(densidades)

        resultados.append({
            'conjunto': nome,
            'tempo_medio': tempo_medio,
            'valor_medio': valor_medio,
            'desvio_valor': desvio_valor,
            'densidade_media': densidade_media
        })

        print(f"  Valor médio: {valor_medio:.2f} (±{desvio_valor:.2f})")
        print(f"  Tempo médio: {tempo_medio:.2f}s")
        print(f"  Densidade média valor/peso: {densidade_media:.2f}\n")

    return resultados

# Execução principal
if __name__ == "__main__":
    print("RESOLVENDO O PROBLEMA DA MOCHILA COM ALGORITMO GENÉTICO")
    print("=" * 60)

    # Executar análises
    resultados_capacidades = analisar_capacidades_diferentes()
    resultados_conjuntos = analisar_conjuntos_diferentes()

    # Apresentar tabelas resumo
    print("\n" + "="*60)
    print("TABELA RESUMO - VARIAÇÃO DA CAPACIDADE")
    print("="*60)
    print(f"{'Capacidade':<12} {'Valor Médio':<12} {'Tempo Médio (s)':<15} {'Desvio Padrão':<12}")
    print("-" * 60)
    for res in resultados_capacidades:
        print(f"{res['capacidade']:<12} {res['valor_medio']:<12.2f} {res['tempo_medio']:<15.2f} {res['desvio_valor']:<12.2f}")

    print("\n" + "="*60)
    print("TABELA RESUMO - DIFERENTES CONJUNTOS DE ITENS")
    print("="*60)
    print(f"{'Conjunto':<12} {'Valor Médio':<12} {'Tempo Médio (s)':<15} {'Densidade Média':<15}")
    print("-" * 60)
    for res in resultados_conjuntos:
        print(f"{res['conjunto']:<12} {res['valor_medio']:<12.2f} {res['tempo_medio']:<15.2f} {res['densidade_media']:<15.2f}")
