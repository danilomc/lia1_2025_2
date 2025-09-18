Aula 11 LIA
Aluno: Danilo Moreira Cordeiro
Análise do Dataset Car Evaluation
Introdução à Ciência de Dados
Dataset: Car Evaluation Database

Fonte: UCI Machine Learning Repository

Objetivo: Classificar carros em categorias de aceitabilidade (unacc, acc, good, vgood)

Estrutura do Projeto:
Importação das bibliotecas
Carregamento do dataset
Exploração inicial (EDA)
Preparação dos dados
Modelagem
Predições e Avaliação
Conclusões


🎯 PRINCIPAIS DESCOBERTAS:

1. MODELO MAIS EFICAZ:
   → Decision Tree (Label) com acurácia de 98.6%

2. FATORES CRÍTICOS PARA ACEITAÇÃO DE CARROS:
   → Segurança: Fator mais importante (não negociável)
   → Capacidade: Carros que transportam mais pessoas são preferidos
   → Custo: Tanto preço de compra quanto manutenção impactam significativamente

3. RECOMENDAÇÕES PARA FABRICANTES:
   → Priorizar segurança em todos os modelos
   → Otimizar capacidade de passageiros
   → Balancear custo inicial vs. custo de manutenção
   → Considerar porta-malas e número de portas como diferenciais

4. INSIGHTS TÉCNICOS:
   → Algoritmos baseados em árvore são ideais para dados categóricos
   → Label Encoding adequado para features ordinais
   → One-Hot Encoding é preferível para algoritmos baseados em distância (KNN)
   → Dataset hierárquico reflete bem decisões do mundo real
   → Performance similar entre métodos de encoding para modelos de árvore

✅ Este modelo pode ser usado para:
   • Avaliação automática de carros em concessionárias
   • Sistema de recomendação para compradores
   • Análise de mercado automotivo
   • Tomada de decisão em desenvolvimento de produtos
