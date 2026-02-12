# Checklist de Validação Final

1. Conferir confrontos base: `WE x TES`, `LNG x TT`, `GEN.G x T1`.
2. Validar ML Mapa e Série.
3. Confirmar que salvar linhas sem editar não altera odd.
4. Validar Consistência: Kills, Torres, Tempo com linhas da sessão.
5. Validar ML + Totais: `Odd (Modelo)`, `Odd (Hist)`, `Odd (Mix)`.
6. Validar Players: auto-preencher, analisar, médias e odds.
7. Validar Campeões: draft colado, lane matchups, resumo por campeão.
8. Conferir `ml_trace_last.json`: `p_map_used`, `resumo_markets_trace`, `players_trace`.
9. Executar manutenção quando sinalizado: retreino, parâmetros, walk-forward.
10. Revisar textos visuais (sem caracteres quebrados).
11. Rodar testes: `python -m unittest discover -s mlcore/tests -p "test_*.py"`.
12. Salvar snapshot dos arquivos críticos antes de entregar.

