# Agent System Definition

## System Agents

### Quality Gate Agent
- **Type**: Validator
- **Priority**: Critical
- **Tools**:
  - `pmat_analyze_complexity`
  - `pmat_detect_satd`
  - `pmat_security_scan`

### Refactoring Agent
- **Type**: Transformer
- **Priority**: High
- **Tools**:
  - `pmat_refactor_code`
  - `pmat_apply_patterns`

### Analysis Agent
- **Type**: Analyzer
- **Priority**: Normal
- **Tools**:
  - `pmat_analyze_code`
  - `pmat_generate_metrics`

## Communication Protocol

- **Message Format**: JSON
- **Transport**: MCP
- **Discovery**: Auto

## Quality Requirements

- **Complexity Limit**: 8
- **Coverage Minimum**: 95%
- **SATD Tolerance**: 0
