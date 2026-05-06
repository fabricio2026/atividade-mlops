# Bank Marketing Subscription Predictor — Documentação Técnica

## Arquitetura da Solução

A solução é dividida em três camadas independentes:

| Camada | Localização | Responsabilidade |
|--------|-------------|-----------------|
| Treinamento | `scripts/train_model.py` | Download, pré-processamento, treino e serialização |
| Serviço | `app/` | Carregamento do modelo e exposição via API REST |
| Testes | `tests/` | Validação unitária e de integração |

### Fluxo de Dados

```
[UCI Repository]
      |
      v
train_model.py --> ColumnTransformer + RandomForest --> models/bank_marketing_model.joblib
                                                                  |
                                                                  v
[Cliente HTTP] --> POST /predict --> Pydantic v2 --> model.predict() --> PredictionResponse
```

---

## Decisões Técnicas

### Algoritmo: RandomForestClassifier

O dataset Bank Marketing é desbalanceado (~88% "no", ~12% "yes"). O RandomForest foi escolhido porque:
- Lida bem com desbalanceamento sem ajustes extras.
- É robusto a outliers (como o campo `duration=0`).
- Não exige normalização numérica, simplificando o pré-processamento.

### Pré-processamento: OrdinalEncoder

Por ser um modelo baseado em árvore, o RandomForest não precisa de one-hot encoding. O `OrdinalEncoder` é mais eficiente em memória. O parâmetro `handle_unknown="use_encoded_value"` garante que categorias desconhecidas em produção não causem erro.

### Pipeline Scikit-Learn

O `Pipeline(preprocessor + classifier)` serializa o pré-processamento junto com o modelo. Isso garante **simetria treino-serving**: a transformação aplicada no predict é exatamente a mesma do fit, eliminando training-serving skew.

### Singleton de Modelo

O modelo é carregado do disco uma única vez via `lifespan` do FastAPI. Chamadas subsequentes ao `/predict` reutilizam o objeto em memória, sem I/O por requisição.

---

## Validação de Entrada (Pydantic v2)

| Mecanismo | Aplicação |
|-----------|-----------|
| `ConfigDict(extra="forbid")` | Rejeita campos desconhecidos com HTTP 422 |
| `Field(ge=..., le=...)` | Valida ranges numéricos (ex.: `age` 18–100, `pdays` 0–999) |
| `Literal[...]` | Restringe categóricas ao conjunto de treinamento |

---

## Estratégia de Testes

### Unitários (`test_schemas.py`)
Validam o contrato Pydantic isoladamente, sem subir a API:
- Entrada válida é aceita.
- Campo numérico fora do range levanta `ValidationError`.
- Valor categórico inválido levanta `ValidationError`.
- Campo extra é rejeitado com `ValidationError`.

### Integração (`test_api.py`)
Sobem a API com `TestClient` e verificam o comportamento real dos endpoints:
- `GET /health` retorna 200 sempre.
- `POST /predict` com payload válido retorna predição (skip se modelo ausente).
- `POST /predict` com payload inválido retorna 422.
- `POST /predict` quando modelo não carregado retorna 503.
- `GET /info` quando modelo não carregado retorna 503.

---

## Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo (necessário antes de iniciar a API)
python scripts/train_model.py

# Rodar os testes
pytest tests/ -v

# Iniciar a API
uvicorn app.main:app --reload
```

## Docker

```bash
# Treinar localmente antes do build (o artefato é copiado para a imagem)
python scripts/train_model.py

docker build -t bank-marketing-api .
docker run -p 8000:8000 bank-marketing-api
```

Acesse `http://localhost:8000/docs` para a documentação interativa (Swagger UI).
