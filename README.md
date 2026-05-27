# Profiler de Escalonamento de Memória para LLMs (Scaling Curve Profiler)

<div align="justify">

## 1. Visão Geral do Projeto

Este projeto implementa um profiler analítico de escalonamento de memória para modelos de linguagem (LLMs), com o objetivo de analisar como o consumo de memória cresce conforme o comprimento da sequência de entrada aumenta.

Em vez de medir diretamente o uso de GPU, o sistema utiliza um modelo teórico baseado na arquitetura Transformer para estimar o consumo de memória de cada componente.

O foco principal é estudar o comportamento assintótico da arquitetura:

* crescimento linear O(n)
* crescimento quadrático O(n²)

</div>

---

## 2. Estrutura Geral do Código

<div align="justify">

O código é dividido em quatro partes principais:

1. Carregamento do modelo e extração de metadados arquiteturais
2. Definição dos parâmetros estruturais do Transformer
3. Modelo analítico de estimativa de memória
4. Geração e plot da curva de escalonamento

Cada etapa representa uma abstração diferente do custo computacional de um LLM.

</div>

---

## 3. Carregamento do Modelo

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16
)
```

<div align="justify">

Neste bloco o modelo TinyLlama é carregado em modo quantizado (4-bit), reduzindo drasticamente o uso de memória real.

O modelo não é utilizado para inferência completa neste experimento. Ele serve apenas para fornecer metadados estruturais como:

* número de cabeças de atenção
* dimensão do embedding (hidden size)
* configuração da arquitetura Transformer

</div>

---

## 4. Extração de Metadados do Transformer

```python
H = model.config.num_attention_heads
hidden = model.config.hidden_size
bytes_per_elem = 2
```

<div align="justify">

Aqui são extraídos os parâmetros fundamentais da arquitetura:

* H representa o número de heads de atenção
* hidden representa a dimensão do espaço latente do modelo
* bytes_per_elem define o custo de memória em precisão FP16

Esses valores são usados para estimar o consumo de memória de forma analítica.

</div>

---

## 5. Modelo Analítico de Memória

```python
def memory_estimate(T):
```

<div align="justify">

Esta função é o núcleo do profiler. Ela estima o consumo de memória para um dado comprimento de sequência T.

O cálculo é dividido em quatro componentes principais.

</div>

---

### 5.1 Projeções QKV (crescimento linear)

```python
mem_qkv = 3 * B * T * hidden * bytes_per_elem
```

<div align="justify">

Representa as projeções de Query, Key e Value.

O crescimento é linear com o tamanho da sequência.

</div>

---

### 5.2 MLP (crescimento linear)

```python
mem_mlp = B * T * (4 * hidden) * bytes_per_elem
```

<div align="justify">

O MLP expande a dimensionalidade interna do modelo em aproximadamente 4 vezes.

Esse componente também cresce linearmente com T.

</div>

---

### 5.3 Matriz de Atenção (crescimento quadrático)

```python
mem_attn = B * H * T * T * bytes_per_elem
```

<div align="justify">

Este é o componente mais crítico do Transformer.

A matriz de atenção possui dimensão T × T para cada head, resultando em crescimento quadrático.

Este é o principal gargalo para contextos longos.

</div>

---

### 5.4 Stream Residual de Saída

```python
mem_out = B * T * hidden * bytes_per_elem
```

<div align="justify">

Representa o fluxo residual entre camadas.

Possui crescimento linear e impacto menor no consumo total.

</div>

---

## 6. Geração da Curva de Escalonamento

```python
seq_lengths = list(range(8, 2049, 32))
```

<div align="justify">

O experimento avalia múltiplos tamanhos de sequência, variando de 8 até 2048 tokens.

Isso permite observar como o custo de memória evolui conforme o contexto aumenta.

</div>

---

## 7. Visualização dos Resultados

```python
plt.plot(seq_lengths, attn_curve, label="Attention (O(n²))")
```

<div align="justify">

O gráfico final compara os diferentes componentes da arquitetura:

* QKV e MLP com crescimento linear
* Atenção com crescimento quadrático
* Memória total combinada

Isso evidencia o impacto dominante da atenção em contextos longos.

</div>

---

## 8. Conclusão

<div align="justify">

O profiler demonstra que o comportamento de memória em Transformers é governado por dois regimes principais:

* componentes lineares (QKV, MLP, residual)
* componente quadrático (atenção)

Na prática, o custo de atenção define o limite de escalabilidade de contexto em LLMs tradicionais.

Esse tipo de análise é fundamental para justificar otimizações como FlashAttention, atenção esparsa e mecanismos de cache.

</div>
