# Arquitetura - gabineteLOLweb3.1

## Objetivo
Aplicativo Streamlit para precificacao de mercados de LoL com:
- ML de vencedor (mapa/serie),
- mercados de totais/handicaps/ML+Totais,
- analise por players,
- trilha de auditoria (`ml_trace_last.json`).

## Camadas

### 1) UI e orquestracao
- Arquivo principal: `plays_app.py`
- Responsabilidades:
  - fluxo de tela (`select -> lines -> results`),
  - widgets e navegacao por abas,
  - gates de "analisar" para evitar recalculo involuntario,
  - chamada dos motores (`app_core/*`, `mlcore/*`),
  - montagem e persistencia de traces.

### 2) Core de app (dominio/estado)
- `app_core/app_state.py`
  - defaults, presets, bootstrap de estado.
- `app_core/settings_core.py`
  - carregar/salvar config, acoes pendentes de preset/reset.
- `app_core/settings_ui.py`
  - UI de parametros/presets.
- `app_core/fusion_core.py`
  - fusao Team x Players com guardiao de coerencia.
- `app_core/trace_core.py`
  - payload combinado de auditoria.
- `app_core/players_core.py`, `app_core/resumo_core.py`, `app_core/odds_core.py`
  - logica de dominio para reduzir acoplamento com Streamlit.

### 3) Motor estatistico/modelo
- `mlcore/*`
  - calibracao,
  - precificacao,
  - walk-forward,
  - treino e utilitarios.

## Fluxo principal

1. Seleciona temporada/times.
2. Preenche linhas.
3. Vai para resultados (`flow_stage=results`).
4. Gate de assinatura valida se parametros mudaram:
   - se mudou: exige clique em "Analisar".
   - se nao mudou: usa analise atual.
5. Gera:
   - odds de ML,
   - mercados (Modelo + Historico/Laplace + Mix),
   - traces (`_ml_trace_last`, `_ml_trace_markets`, `_ml_trace_players`).
6. Persiste trace combinado em `ml_trace_last.json`.

## Trace (auditoria)

Fontes internas:
- `_ml_trace_last` (base ML),
- `_ml_trace_markets` (Resumo),
- `_ml_trace_players` (Players).

Arquivo final:
- `ml_trace_last.json` via `core_build_combined_trace(...)`.

Campos importantes:
- `p_map_used`,
- bloco de fusao (`p_team`, `p_players`, `p_fused`, pesos, cobertura),
- listas de picks por mercado/players.

## Regras de qualidade

- Nao duplicar regra de negocio em UI.
- Qualquer ajuste de peso/fusao deve entrar em `app_core/*`.
- Qualquer novo output critico deve entrar no trace combinado.
- Mudancas em odd/fusao exigem teste em `mlcore/tests`.

## Suite de testes

- Pasta: `mlcore/tests`
- Cobertura atual:
  - estado/presets/settings,
  - odds/resumo/players,
  - fusao e guardioes,
  - trace combinado,
  - regressao de odds,
  - smoke.

Comando:
- `python -m unittest discover -s mlcore/tests -p "test_*.py"`

## Roadmap tecnico (curto)

1. Consolidar ultimos blocos grandes de `plays_app.py` em `app_core/*`.
2. Adicionar regressao fixture para cenarios sensiveis (LNG/TT, WE/TES, GEN/T1).
3. Finalizar higienizacao de strings/encoding na UI.
4. Incluir smoke de navegacao por abas criticas com asserts de render minimo.

