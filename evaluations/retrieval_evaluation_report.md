# Informe de evaluación de la recuperación semántica

**Proyecto:** AI Research Assistant  
**Etapa:** Entrega 2 — Recuperación semántica  
**Versión de la evaluación:** 1.0  
**Fecha:** 16 de noviembre de 2026

## 1. Objetivo

Evaluar el funcionamiento del mecanismo de recuperación semántica implementado en el proyecto AI Research Assistant, comprobando su capacidad para localizar fragmentos relevantes de documentación técnica a partir de consultas expresadas en lenguaje natural.

También se analiza su comportamiento ante consultas para las cuales el corpus no contiene información que permita proporcionar una respuesta.

## 2. Entorno de evaluación

La evaluación se realizó utilizando los siguientes componentes:

| Componente | Configuración |
|---|---|
| Generación de embeddings | Ollama |
| Modelo de embeddings | nomic-embed-text:latest |
| Dimensión de los vectores | 768 |
| Métrica de similitud | Similitud coseno |
| Recuperador | ai_research_assistant.retrieval |
| Valor de top_k | 4 |
| Corpus | documentacion_demo.md |
| Documentos indexados | 1 |
| Fragmentos indexados | 4 |

Se utilizó el índice documental y el índice vectorial generados durante las etapas anteriores del proyecto.

## 3. Metodología

Se diseñó un conjunto inicial de nueve consultas, clasificadas en dos categorías:

- **Consultas positivas:** preguntas cuya información está presente en los fragmentos indexados.
- **Consultas negativas:** preguntas cuya información no está disponible en el corpus de evaluación.

Para cada consulta se ejecutó el método `search()` del módulo `retrieval.py`, recuperando los cuatro fragmentos con mayor similitud semántica.

### 3.1. Métricas utilizadas

**Recall@k:** mide la proporción de fragmentos relevantes que aparecen entre los primeros k resultados recuperados.

**Mean Reciprocal Rank (MRR):** mide la posición del primer fragmento relevante. Cuanto más cerca de la primera posición aparezca, mayor será el valor de esta métrica.

Las métricas agregadas se calcularon utilizando exclusivamente las cinco consultas positivas.

## 4. Resultados

### 4.1. Resultados de las consultas positivas

| ID | Consulta | Recall@1 | Recall@3 | Recall@4 | RR |
|---|---|---:|---:|---:|---:|
| P1 | Dirección HTTP de Ollama | 1.00 | 1.00 | 1.00 | 1.00 |
| P2 | Endpoint para generar texto | 1.00 | 1.00 | 1.00 | 1.00 |
| P3 | Endpoint para obtener representaciones vectoriales | 1.00 | 1.00 | 1.00 | 1.00 |
| P4 | Identificación del documento de origen | 0.50 | 1.00 | 1.00 | 1.00 |
| P5 | Utilidad de los offsets de caracteres | 0.50 | 1.00 | 1.00 | 1.00 |

### 4.2. Métricas agregadas

| Métrica | Resultado |
|---|---:|
| Recall@1 | 0.8000 |
| Recall@3 | 1.0000 |
| Recall@4 | 1.0000 |
| MRR | 1.0000 |

Los resultados muestran que todas las consultas positivas recuperaron al menos un fragmento relevante en la primera posición.

En P4 y P5 se identificaron dos fragmentos relevantes para cada consulta. El primer resultado contenía uno de ellos y el segundo resultado contenía el otro. Por esta razón, Recall@1 fue de 0.5 en ambos casos, aunque el primer fragmento recuperado fuera correcto.

Al considerar los tres primeros resultados, se recuperó la totalidad de los fragmentos etiquetados como relevantes para las cinco consultas.

### 4.3. Resultados de las consultas negativas

Se evaluaron cuatro preguntas cuya respuesta no se encuentra en el corpus:

| ID | Consulta | Score del primer resultado |
|---|---|---:|
| N1 | ¿Cómo se configura un router TP-Link? | 0.5630 |
| N2 | ¿Qué algoritmo utiliza Ollama para entrenar sus modelos? | 0.6112 |
| N3 | ¿Cómo instalar CUDA en Windows? | 0.5349 |
| N4 | ¿Qué sistema operativo necesita Ollama para ejecutarse? | 0.5890 |

En todos los casos, el recuperador devolvió fragmentos, a pesar de que el corpus no contenía evidencia suficiente para responder a las preguntas.

Este comportamiento es propio de la recuperación vectorial: el mecanismo ordena los fragmentos disponibles según su similitud con la consulta, pero no determina por sí mismo si estos contienen información suficiente para responder.

También se observó que las puntuaciones de similitud de las consultas negativas se solapan con las obtenidas en las consultas positivas. Por tanto, los resultados no justifican establecer un umbral fijo de similitud como único criterio para decidir si una pregunta puede responderse.

## 5. Análisis

La evaluación permite identificar los siguientes aspectos:

1. El mecanismo recupera correctamente la evidencia asociada a las cinco consultas positivas.
2. La recuperación de toda la evidencia relevante se alcanza con un valor de `top_k` igual a 3 en este conjunto de pruebas.
3. La primera posición contiene evidencia relevante para todas las consultas positivas, lo que se refleja en un MRR de 1.0.
4. Las consultas negativas también generan resultados, por lo que la recuperación semántica no debe confundirse con la validación de suficiencia de evidencia.
5. La similitud coseno, de forma aislada, no permite distinguir de manera fiable entre consultas respondibles y no respondibles en este corpus.

## 6. Limitaciones

Esta evaluación constituye una prueba inicial y controlada, no una validación general del sistema.

Sus principales limitaciones son:

- El corpus contiene un único documento.
- Se dispone de solamente cuatro fragmentos indexados.
- El conjunto de evaluación contiene nueve consultas.
- Las consultas y los fragmentos relevantes fueron definidos manualmente.
- No se ha evaluado todavía el comportamiento con documentación técnica extensa, heterogénea o con múltiples documentos.
- No se ha implementado un mecanismo específico para determinar la suficiencia de la evidencia recuperada.

Por tanto, los resultados deben interpretarse dentro del alcance de este corpus y no como una medida definitiva de la calidad del recuperador en escenarios reales.

## 7. Conclusiones

El mecanismo de recuperación semántica implementado cumple satisfactoriamente su objetivo inicial dentro del conjunto de pruebas utilizado.

Se obtuvo un Recall@3 de 1.0 y un MRR de 1.0, lo que demuestra que las cinco consultas positivas recuperaron toda la evidencia etiquetada como relevante entre los tres primeros resultados y que la primera evidencia relevante apareció siempre en la posición inicial.

Sin embargo, las pruebas negativas evidenciaron que el recuperador devuelve resultados incluso cuando no existe información suficiente para responder.

En consecuencia, se considera completada la evaluación inicial del mecanismo de recuperación semántica y se identifica como siguiente objetivo técnico la incorporación de un mecanismo de validación de suficiencia de evidencia antes de generar respuestas.

## 8. Artefactos asociados

- `retrieval_dataset.json`: conjunto de consultas y fragmentos relevantes.
- `evaluate_retrieval.py`: script para ejecutar la evaluación y calcular las métricas.
- `retrieval_evaluation_report.md`: informe técnico de resultados.

Estos artefactos permiten conservar la configuración de la prueba, reproducir la evaluación y comparar los resultados con futuras versiones del sistema.