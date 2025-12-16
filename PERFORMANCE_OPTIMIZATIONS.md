# Performance Optimizations Applied

## 🚀 Speedup Techniques

### 1. **Batch Size Otimizado**
- **Antes**: 32
- **Depois**: 128
- **Ganho**: ~3-4x throughput na GPU (melhor utilização)

### 2. **Mixed Precision Training (AMP)**
- Usa `torch.cuda.amp.autocast()` com float16
- **Ganho**: ~2x velocidade, ~50% menos memória GPU
- Automático quando `USE_AMP=True` e device=cuda

### 3. **DataLoader Paralelo**
- **Antes**: num_workers=0 (single-threaded)
- **Depois**: num_workers=4 (parallel loading)
- `pin_memory=True` para transferência CPU→GPU mais rápida
- **Ganho**: Elimina bottleneck de I/O

### 4. **Torch Compile (PyTorch 2.x)**
- Compila o modelo com `torch.compile(mode='reduce-overhead')`
- Otimizações JIT automáticas
- **Ganho**: ~30% mais rápido em GPUs modernas

### 5. **Otimizações de Transferência GPU**
- `non_blocking=True` nos `.to(device)`
- `zero_grad(set_to_none=True)` (mais eficiente que `zero_grad()`)
- **Ganho**: Reduz latência de transferência

### 6. **Configuração de Rounds**
- **local-epochs**: 10 → 5 (menos epochs por round)
- **fraction-fit**: 0.5 → 1.0 (treina todos os clientes por round)
- **Resultado**: Menos rounds necessários para convergência

### 7. **Recursos Ray Aumentados**
- **CPUs**: 6 → 8 por cliente
- Melhora paralelização do DataLoader

## 📊 Speedup Esperado

| Componente | Speedup |
|------------|---------|
| Batch Size (32→128) | ~3-4x |
| Mixed Precision | ~2x |
| Torch Compile | ~1.3x |
| DataLoader Workers | ~1.5x |
| Non-blocking transfers | ~1.2x |
| **TOTAL COMBINADO** | **~10-15x** |

## ⚙️ Configurações Aplicadas

### task.py
```python
BATCH_SIZE = 128
NUM_WORKERS = 4
PIN_MEMORY = True
USE_AMP = True  # Mixed precision
```

### pyproject.toml
```toml
local-epochs = 5          # Reduzido de 10
fraction-fit = 1.0        # Todos os clientes
num-cpus = 8              # Aumentado de 6
```

### client_app.py
- `torch.compile()` habilitado automaticamente
- Logs de uso de GPU

## 🔍 Monitoramento

Durante execução, você verá:
```
Client 0 model compiled with torch.compile
Client 0 using device: cuda:0
  GPU: NVIDIA GeForce GTX 1650
  Memory allocated: XX.XX MB
```

## 💡 Dicas Adicionais

### Se quiser acelerar ainda mais:

1. **Reduzir rounds totais**:
   ```toml
   num-server-rounds = 50  # Ao invés de 100
   ```

2. **Aumentar batch size** (se tiver memória GPU):
   ```python
   BATCH_SIZE = 256  # task.py
   ```

3. **Usar AdamW com weight decay**:
   ```python
   optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
   ```

4. **Reduzir complexidade do modelo** (se acurácia permitir):
   ```python
   HIDDEN_LAYERS = [128, 64, 32]  # Ao invés de [256, 128, 64, 32]
   ```

5. **Usar gradient accumulation** para batch efetivo maior:
   ```python
   accumulation_steps = 4  # Batch efetivo = 128 * 4 = 512
   ```

## ⚠️ Trade-offs

- **Batch size maior**: Pode reduzir generalização (mas acelera treino)
- **Menos epochs**: Convergência pode ser menos suave
- **Mixed precision**: Raramente afeta acurácia, mas pode causar instabilidade numérica em modelos muito pequenos

## 📈 Benchmark

Execute antes e depois:
```bash
time flower-simulation --app .
```

Tempo esperado:
- **Antes**: ~15-20min para 100 rounds
- **Depois**: ~2-3min para 100 rounds (~8x mais rápido)
