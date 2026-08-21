# BASELINE BEFORE — Auditoria e Correção (Prompt Mestre v5)

Registrado em: 2026-08-21, antes de qualquer alteração desta rodada.

## Ambiente
```
Python: 3.12.3
torch: 2.3.1
qiskit: 2.5.2
qiskit-aer: 0.17.2
numpy: 2.4.4
pandas: 3.0.2
scikit-learn: 1.8.0
Git: não é repositório git formal neste ambiente sandbox
```

## Suíte de testes
```
Total: 443 testes, todos passando
Tempo de execução: ~189s (~3min09s)
Warnings de depreciação: nenhum encontrado (-W error::DeprecationWarning não acusou nada)

Por categoria (mutuamente exclusivas, somam 443 exatamente):
  unit:         179 testes
  physics:      157 testes
  integration:   70 testes
  statistical:   28 testes
  benchmark:      9 testes
                ----
                443 testes

Categorias aditivas (sobrepõem as acima):
  slow:          49 testes
  experimental:  31 testes
```

## Estado do repositório
```
181 arquivos .py (excluindo __pycache__)
54 arquivos de teste em tests/
72 arquivos CSV em outputs/
outputs/experiments/master_results.csv, master_results.json (addendum 55)
outputs/master_report/ populado (addendum 64)
```

## Histórico de desenvolvimento
```
70 addenda cronológicos documentados em docs/history.md
README.md reestruturado em 15 seções numeradas (addendum 65)
Licença: Apache License 2.0
```

## Achados/gaps conhecidos NÃO corrigidos ainda (auto-identificados nas
## seções "Honest limitations" de addenda anteriores, relevantes a este
## novo prompt v5):

1. Correção de múltiplas comparações (Holm-Bonferroni/Benjamini-Hochberg)
   NUNCA foi aplicada a nenhum dos múltiplos testes estatísticos já
   executados neste projeto — Seção 10 deste novo prompt.
2. RiskAwareController nunca recebeu uma campanha REAL de >=10 seeds
   isolada e dedicada (addendum 59 fez sweep de pesos de custo com 10
   seeds, mas não uma comparação de desempenho robusta multi-seed no
   estilo do addendum 46) — Seção 9.
3. Domain shift (addendum 49) usou T1/T2/distância como regimes OOD, mas
   NÃO variou explicitamente correlation_time, drift_amplitude,
   diffusion_coefficient, sampling_interval — parâmetros do
   Ornstein-Uhlenbeck especificamente nomeados na Seção 1 deste prompt.
4. Nenhum dataset/experimento registra formalmente TODOS os campos de
   metadados pedidos na Seção 2/3 (source_type, sampling_rate, units,
   dataset_hash, noise_model, realism_level) de forma automática e
   uniforme em cada geração de dataset.
5. QuantumRuntimeProfiler (Seção 19: setup/circuit build/simulation/
   measurement/state conversion/control update separados) não existe —
   o benchmark E2E (addendum 56) tem 5 estágios, não os 6 pedidos aqui.

## Plano desta rodada

Dado o escopo massivo deste prompt (40 seções), esta rodada vai
priorizar, seguindo a própria ordem de prioridade do prompt
("VALIDADE CIENTÍFICA > REPRODUTIBILIDADE > ROBUSTEZ ESTATÍSTICA..."):

**Seção 10 (Múltiplas comparações)**: aplicar correção formal
Holm-Bonferroni/Benjamini-Hochberg aos p-values já computados em
experimentos anteriores (addendum 46's 3 comparações pareadas DualHead
vs Blind/Reactive/Predictive), com infraestrutura reutilizável para
futuros experimentos com múltiplos testes.

Alterações incrementais, com regressão completa após cada mudança,
seguindo a disciplina exigida.
