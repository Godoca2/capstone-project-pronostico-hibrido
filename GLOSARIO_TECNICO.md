# 📚 GLOSARIO TÉCNICO Y CIENTÍFICO
## Proyecto: Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile

**Versión:** 1.0  
**Fecha:** 23 de Noviembre de 2025  
**Autor:** César Godoy Delaigue

---

## 🔤 ÍNDICE ALFABÉTICO

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w)

---

## A

### Activación (Función de)
Función matemática aplicada a la salida de una neurona en una red neuronal que introduce no linealidad. Las más comunes son ReLU, sigmoid, tanh y linear.

**Ejemplo en el proyecto:**
```python
layers.Conv2D(64, 3, activation='relu')
```

### AE (Autoencoder)
Red neuronal diseñada para aprender representaciones comprimidas (embeddings) de datos. Consta de un encoder (compresor) y un decoder (reconstructor).

**Aplicación:** Compresión espacial de campos de precipitación de 6437 → 64 dimensiones.

### Agregación Temporal
Proceso de combinar datos de alta frecuencia temporal en intervalos más largos (ej: datos horarios → diarios mediante suma o promedio).

**Ejemplo:** ERA5 horario (8784 horas/año) → ERA5 diario (366 días/año).

### Anisotropía
Propiedad de un campo espacial donde la correlación depende de la dirección. Opuesto a isotropía.

**En precipitaciones:** Correlación mayor en dirección norte-sur que este-oeste debido a topografía andina.

---

## B

### Baseline (Modelo)
Modelo simple de referencia usado para comparación. Debe superarse para demostrar utilidad de modelos complejos.

**Baselines del proyecto:**
- **Persistence:** Pronóstico = último valor observado
- **Climatología:** Pronóstico = promedio histórico

### Batch Normalization
Técnica de normalización de activaciones entre capas de una red neuronal que estabiliza y acelera el entrenamiento.

**Fórmula:**
```
BN(x) = γ * (x - μ) / √(σ² + ε) + β
```

### Batch Size
Número de muestras procesadas simultáneamente en una iteración de entrenamiento de red neuronal.

**Valor usado:** 16 muestras por batch.

---

## C

### Campo Aleatorio (Random Field)
Variable que toma valores en cada punto del espacio de forma estocástica. Las precipitaciones son un campo aleatorio espacio-temporal.

### CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
Dataset satelital de precipitación a 0.05° de resolución, usado para validación cruzada.

**Cobertura:** Global, 1981-presente, actualización cuasi-tiempo real.

### Climatología
Promedio de largo plazo (típicamente 30 años) de una variable climática para cada día/mes del año.

**Uso como baseline:** Pronóstico = precipitación promedio para esa fecha del año.

### Convolución (Convolutional Layer)
Operación que aplica filtros (kernels) a una ventana espacial de datos para extraer características locales.

**Ejemplo:**
```python
Conv2D(filters=64, kernel_size=3, padding='same')
```

### Conv2DTranspose
Operación de convolución "inversa" usada para upsampling en decoders. Proyecta de baja a alta resolución espacial.

**Ventaja sobre UpSampling2D:** Implementación determinista compatible con GPUs.

### Covarianza Espacial
Medida de co-variación entre valores de un campo en dos ubicaciones separadas por una distancia h.

**Relacionado:** Variograma, función de covarianza, correlación espacial.

---

## D

### Data Augmentation
Técnicas para aumentar artificialmente el tamaño del dataset mediante transformaciones (rotaciones, traslaciones, ruido).

**No usado en el proyecto:** Preservamos estructura espacial real de precipitaciones.

### Dataset
Conjunto de datos usado para entrenamiento, validación y prueba de modelos.

**Splits del proyecto:**
- Train: 70% (251 secuencias)
- Validation: 15% (53 secuencias)
- Test: 15% (55 secuencias)

### Decoder
Componente del autoencoder que reconstruye los datos originales desde la representación latente comprimida.

**Arquitectura:** Conv2DTranspose → BatchNorm → Upsampling (×3 bloques).

### Determinismo
Propiedad de un algoritmo que produce resultados idénticos con las mismas entradas y configuración.

**Implementación:**
```python
SEED = 42
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

### Dilated Convolution (Convolución Dilatada)
Convolución con "huecos" entre píxeles del kernel, permitiendo receptive fields grandes sin aumentar parámetros.

**Dilations usadas:** [1, 2, 4, 8] para alcanzar RF ≈ 40 celdas.

### DMD (Dynamic Mode Decomposition)
Técnica de análisis de sistemas dinámicos que descompone series temporales en modos oscilatorios con frecuencias y tasas de crecimiento/decaimiento.

**Aplicación:** Proyección temporal de embeddings latentes del autoencoder.

### Dropout
Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante entrenamiento para prevenir overfitting.

**No usado en el proyecto:** Preferimos regularización L2 por interpretabilidad.

---

## E

### Early Stopping
Estrategia que detiene el entrenamiento cuando la métrica de validación deja de mejorar por un número de épocas (patience).

**Configuración:** Patience = 15 épocas, restore_best_weights = True.

### ECMWF (European Centre for Medium-Range Weather Forecasts)
Centro europeo que produce reanálisis ERA5, considerado gold standard en datos meteorológicos.

### EDA (Exploratory Data Analysis)
Análisis exploratorio de datos para entender distribuciones, patrones, anomalías y relaciones antes del modelado.

**Notebook 01:** EDA espacio-temporal de precipitaciones.

### Embedding (Representación Latente)
Representación comprimida de alta dimensión de datos en un espacio de menor dimensión que preserva estructura semántica.

**Dimensión latente:** 64 valores por snapshot espacial.

### Encoder
Componente del autoencoder que comprime datos de alta dimensión a representación latente de baja dimensión.

**Arquitectura:** Conv2D dilatadas → MaxPooling (×3) → Flatten → Dense.

### Época (Epoch)
Una pasada completa del algoritmo de entrenamiento sobre todo el dataset.

**Entrenamiento:** 100 épocas (stopped en 97 por early stopping).

### ERA5 (ECMWF Reanalysis v5)
Dataset de reanálisis global horario a 0.25° de resolución (1940-presente) que combina observaciones con modelos físicos.

**Variables usadas:** Total Precipitation (tp) en metros.

### Error Cuadrático Medio → Ver **MSE**

### Estacionalidad
Patrón recurrente en series temporales con periodo anual (4 estaciones).

**En Chile:**
- Norte: Máximo verano (lluvias altiplánicas)
- Centro: Máximo invierno (frentes fríos)
- Sur: Distribuido todo el año (océano)

### Evento Extremo
Evento de precipitación que excede un umbral estadístico (típicamente percentil 95 o 99).

**Definición en proyecto:** Precipitación ≥ 10 mm/día (6.2% de píxeles).

---

## F

### Feature Map
Salida de una capa convolucional que representa características aprendidas de los datos de entrada.

**Ejemplo:** Primera capa extrae 32 feature maps de bordes/texturas.

### Forecast Horizon (Horizonte de Pronóstico)
Tiempo futuro para el cual se realiza una predicción.

**Horizontes evaluados:** 1, 3 y 7 días adelante.

### Forecasting
Proceso de predecir valores futuros de una variable basándose en observaciones pasadas.

### Función de Covarianza
Función que describe la covarianza entre dos puntos separados por un vector h.

**Relación con variograma:**
```
C(h) = C(0) - γ(h)
```

---

## G

### Gaussian (Modelo)
Modelo de variograma con transición suave sin punto de inflexión.

**Ecuación:**
```
γ(h) = nugget + sill * [1 - exp(-(h/range)²)]
```

### Geoestadística
Rama de la estadística espacial que estudia variables regionalizadas con correlación espacial, desarrollando métodos como kriging y variogramas.

**Aplicación:** Diseño de arquitectura CNN y loss function ponderada.

### GPU (Graphics Processing Unit)
Procesador especializado en cálculos paralelos, esencial para entrenar redes neuronales profundas.

**GPU usada:** NVIDIA compatible con TensorFlow determinista.

### Gradiente (Gradient)
Vector de derivadas parciales que indica la dirección de máximo crecimiento de una función.

**En deep learning:** Usado por backpropagation para actualizar pesos.

### Grid (Grilla Espacial)
Discretización regular del espacio en celdas rectangulares.

**Grid ERA5:** 157 latitudes × 41 longitudes (0.25° resolución).

### Grid Search
Método de optimización exhaustiva que prueba todas las combinaciones de hiperparámetros en un grid predefinido.

**Notebook 06:** 13 configuraciones planificadas.

---

## H

### Hiperparámetro
Parámetro del modelo que se fija antes del entrenamiento (no es aprendido por el algoritmo).

**Ejemplos:** learning_rate, batch_size, latent_dim, epochs.

### Horizonte → Ver **Forecast Horizon**

---

## I

### Isotropía
Propiedad de un campo espacial donde la correlación depende solo de la distancia (no de la dirección).

**Asunción en variograma:** Simplifica modelado, aproximación razonable para precipitaciones.

### Interpolación Espacial
Proceso de estimar valores en ubicaciones no observadas basándose en valores cercanos conocidos.

**Método usado:** Kriging Ordinario (óptimo bajo supuestos gaussianos).

---

## K

### Keras
API de alto nivel para construir redes neuronales, integrada en TensorFlow 2.x.

**Ventajas:** Sintaxis simple, callbacks, integración con TensorBoard.

### Kernel (en CNN)
Matriz pequeña de pesos que se desliza sobre la entrada para realizar convolución y extraer características.

**Tamaños usados:** 3×3 (estándar para capturar patrones locales).

### KoVAE (Kolmogorov-Arnold Variational Autoencoder)
Variante de VAE que usa teorema de Kolmogorov-Arnold para representaciones más interpretables.

**Notebook 05:** Implementación exploratoria comparativa.

### Kriging
Método geoestadístico de interpolación espacial que minimiza varianza del error de estimación.

**Propiedades:**
- Insesgado: E[Z*(s) - Z(s)] = 0
- Óptimo: Minimiza Var[Z*(s) - Z(s)]
- Provee incertidumbre: Varianza kriging σ²(s)

### Kriging Ordinario
Variante de kriging que asume media desconocida pero constante en el dominio.

**Ecuación del predictor:**
```
Z*(s₀) = Σᵢ λᵢ Z(sᵢ)
Restricción: Σᵢ λᵢ = 1
```

---

## L

### L2 Regularization (Regularización L2)
Penalización añadida a la función de pérdida proporcional a la suma de cuadrados de los pesos.

**Objetivo:** Prevenir overfitting favoreciendo pesos pequeños.

**Fórmula:**
```
Loss_total = Loss_original + λ * Σ(w²)
```

**λ usado:** 0.0001

### Lag (en variograma)
Distancia de separación h entre pares de puntos usada para calcular semivarianza.

**20 lags usados:** Desde 0° hasta ~10° con intervalos regulares.

### Latent Dimension (Dimensión Latente)
Tamaño del espacio embedding (número de variables latentes).

**Valor usado:** 64 (compresión 100.3x desde 6437 píxeles).

### Learning Rate (Tasa de Aprendizaje)
Hiperparámetro que controla el tamaño del paso en la actualización de pesos durante entrenamiento.

**Valor inicial:** 0.001 (Adam optimizer)  
**Decay:** ReduceLROnPlateau con factor=0.5, patience=7

### Likelihood (Verosimilitud)
Probabilidad de observar los datos dado un modelo con parámetros específicos.

**En variograma:** Maximización de verosimilitud para ajustar parámetros.

### Linear Activation
Función de activación identidad: f(x) = x, usada en capas de salida para regresión.

**Capa output:**
```python
layers.Conv2D(1, 3, activation='linear')
```

### Loss Function (Función de Pérdida)
Métrica que cuantifica la discrepancia entre predicciones y valores reales, minimizada durante entrenamiento.

**Loss usado:** Weighted MSE ponderado por varianza kriging.

### LSTM (Long Short-Term Memory)
Arquitectura de red neuronal recurrente que puede capturar dependencias de largo plazo en secuencias.

**No usado en el proyecto:** Preferimos DMD para dinámica temporal por interpretabilidad.

---

## M

### Macrozona
División geográfica de Chile en regiones climáticas homogéneas.

**3 macrozonas:**
- **Norte:** -17° a -30° (árido)
- **Centro:** -30° a -40° (mediterráneo)
- **Sur:** -40° a -56° (templado oceánico)

### MAE (Mean Absolute Error)
Promedio de valores absolutos de errores de predicción.

**Fórmula:**
```
MAE = (1/n) * Σ|yᵢ - ŷᵢ|
```

**Ventaja:** Interpretable en unidades originales (mm/día).

### MaxPooling
Operación de downsampling que toma el valor máximo en una ventana espacial.

**Configuración:** MaxPooling2D(2,2) reduce dimensiones a la mitad.

### Media Móvil (Moving Average)
Promedio de ventana deslizante usado para suavizar series temporales y resaltar tendencias.

**Ventana usada:** 7 días para visualizar patrones semanales.

### Modelo Esférico (Spherical Model)
Modelo de variograma con crecimiento lineal inicial y plateau en el range.

**Ecuación:**
```
γ(h) = nugget + sill * [1.5(h/r) - 0.5(h/r)³]  si h ≤ r
γ(h) = nugget + sill                            si h > r
```

**Parámetros ajustados:** Range=8.15°, Sill=23.67, Nugget=0.0

### MSE (Mean Squared Error)
Promedio de errores al cuadrado, penaliza más los errores grandes que MAE.

**Fórmula:**
```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

### Multi-step Forecasting
Predicción de múltiples pasos temporales futuros (horizontes 1, 3, 7 días).

**Estrategia:** Proyección DMD con A^h donde h es el horizonte.

---

## N

### NetCDF (Network Common Data Form)
Formato de archivo autodescriptivo para datos científicos multidimensionales (tiempo, lat, lon, variables).

**Archivos usados:**
- `era5_precipitation_chile_full.nc` (45.46 MB)
- `kriging_precipitation_june_2020.nc`

### Normalización
Transformación de datos a escala estándar (típicamente media 0, varianza 1) para mejorar convergencia de redes neuronales.

**Método usado:** StandardScaler (z-score).

### Nugget Effect
Discontinuidad en el origen del variograma que representa variabilidad a escalas menores que la resolución de muestreo o ruido de medición.

**Valor ajustado:** Nugget = 0.0 → Datos limpios sin ruido sub-grid significativo.

---

## O

### Optimizer (Optimizador)
Algoritmo que actualiza pesos de la red neuronal para minimizar la función de pérdida.

**Optimizador usado:** Adam (Adaptive Moment Estimation)
- Combina momentum y RMSprop
- Adaptativo por parámetro
- Robusto a gradientes ruidosos

### Overfitting
Fenómeno donde el modelo aprende ruido y detalles específicos del training set, perdiendo capacidad de generalización.

**Mitigación:**
- Regularización L2
- Early stopping
- Datos de validación independientes

---

## P

### Padding
Estrategia para manejar bordes en convoluciones agregando píxeles extra alrededor de la entrada.

**Tipos:**
- `'same'`: Mantiene dimensiones (usado en el proyecto)
- `'valid'`: Sin padding, reduce dimensiones

### Parámetros (del modelo)
Pesos y sesgos de la red neuronal que son aprendidos durante el entrenamiento.

**Conteo total:** ~2.5M parámetros en el autoencoder.

### Patience (en callbacks)
Número de épocas sin mejora antes de activar early stopping o reducción de learning rate.

**Configuración:**
- EarlyStopping: patience=15
- ReduceLROnPlateau: patience=7

### Percentil
Valor bajo el cual cae un porcentaje dado de observaciones.

**P95 (percentil 95):**
- Norte: 2.10 mm/día
- Centro: 5.78 mm/día
- Sur: 8.82 mm/día

### Persistence (Modelo)
Baseline que asume el futuro será igual al último valor observado.

**Ecuación:**
```
ŷ(t+h) = y(t)  ∀h > 0
```

### Pooling → Ver **MaxPooling**

### Precipitación Total (Total Precipitation)
Suma de precipitación convectiva y estratiforme, variable clave del proyecto.

**Unidades ERA5:** Metros (convertidos a mm/día).

---

## R

### R² (Coeficiente de Determinación)
Proporción de varianza en la variable dependiente explicada por el modelo.

**Fórmula:**
```
R² = 1 - SS_res / SS_tot
```

**R² kriging:** 0.9923 (ajuste excelente).

### Range (Rango en variograma)
Distancia a partir de la cual la semivarianza se estabiliza en el sill (correlación espacial se vuelve despreciable).

**Valor ajustado:** 8.15° (~905 km)

**Interpretación:** Dos puntos separados >8.15° son espacialmente independientes.

### Reanálisis (Reanalysis)
Dataset que combina observaciones históricas con modelos numéricos para producir grids consistentes espacio-temporales.

**ERA5:** Reanálisis de quinta generación del ECMWF.

### Receptive Field (Campo Receptivo)
Región de la entrada que influye en la activación de una neurona específica en capas profundas.

**RF del proyecto:** ~40 celdas (cumple requisito de 33 del variograma).

### Reconstrucción (en autoencoder)
Output del decoder que intenta recuperar la entrada original desde la representación latente.

**Métrica:** MAE entre input y reconstrucción = 0.348 (escala normalizada).

### Reducción de Dimensionalidad
Proceso de proyectar datos de alta dimensión a menor dimensión preservando información importante.

**Técnicas:** PCA (lineal), Autoencoders (no lineal), UMAP.

### Regularización
Técnicas para prevenir overfitting penalizando modelos complejos.

**Tipos usados:**
- L2 regularization (λ=0.0001)
- Early stopping
- Batch normalization

### ReLU (Rectified Linear Unit)
Función de activación no lineal: f(x) = max(0, x).

**Ventajas:** Simple, no saturación para x>0, gradientes no desaparecen.

### Residuo (Residual)
Diferencia entre valor observado y predicho: e = y - ŷ.

**Análisis de residuos:** Diagnóstico de sesgo, heteroscedasticidad, autocorrelación.

### Resolución Espacial
Tamaño de la celda de grid, determina el nivel de detalle espacial.

**ERA5:** 0.25° (~28 km en latitudes medias)
**CHIRPS:** 0.05° (~5.5 km)
**Kriging:** 0.1° (~11 km)

### RMSE (Root Mean Squared Error)
Raíz cuadrada del MSE, tiene las mismas unidades que la variable original.

**Fórmula:**
```
RMSE = √[(1/n) * Σ(yᵢ - ŷᵢ)²]
```

---

## S

### Scaler → Ver **StandardScaler**

### SEED (Semilla Aleatoria)
Valor inicial que determina la secuencia de números pseudoaleatorios, garantiza reproducibilidad.

**Valor usado:** 42 (convención de la comunidad científica).

### Semivarianza (Semivariance)
Mitad de la varianza promedio de diferencias entre pares de puntos separados por distancia h.

**Fórmula:**
```
γ(h) = (1/2N(h)) * Σ[Z(sᵢ) - Z(sᵢ+h)]²
```

### Sequence (Secuencia Temporal)
Ventana deslizante de observaciones consecutivas usada como input del modelo.

**Ventana usada:** 7 días (7 snapshots consecutivos).

### Sill
Valor asintótico del variograma, representa la varianza total del campo.

**Valor ajustado:** 23.67 mm²/día²

**Interpretación:** Varianza máxima entre puntos no correlacionados.

### Skip Connection
Conexión directa que salta capas, permite flujo de gradientes y preserva información.

**No usado explícitamente:** Arquitectura simple encoder-decoder sin ResNet-style skips.

### Snapshot
Estado completo del campo espacial en un instante de tiempo.

**Ejemplo:** Precipitación en 6437 píxeles de Chile en un día específico.

### Spatial Weights (Pesos Espaciales)
Ponderaciones usadas en loss function derivadas de la inversa de la varianza kriging.

**Racionalidad:** Mayor peso en zonas de baja incertidumbre.

### Split (Train/Val/Test)
División del dataset en conjuntos independientes para entrenamiento, validación y prueba.

**Proporciones:** 70% / 15% / 15%

### StandardScaler
Normalizador que transforma datos a media 0 y desviación estándar 1.

**Transformación:**
```
z = (x - μ) / σ
```

### Stride
Paso del desplazamiento del kernel en convoluciones o pooling.

**Stride=2:** Reduce dimensiones a la mitad (downsampling).

### SVD (Singular Value Decomposition)
Factorización matricial: A = UΣVᵀ, usada en DMD para identificar modos dinámicos.

**SVD rank:** Número de valores singulares retenidos (0.99 = 99% varianza).

---

## T

### TensorFlow
Framework de código abierto de Google para machine learning y redes neuronales.

**Versión usada:** 2.10.0 (compatible con determinismo GPU).

### Topografía
Relieve del terreno que influye fuertemente en precipitaciones por forzamiento orográfico.

**Cordillera de los Andes:** Barrera que crea gradiente de precipitación este-oeste.

### Training Loop
Proceso iterativo de forward pass (predicción) + backward pass (gradientes) + actualización de pesos.

### Transpose Convolution → Ver **Conv2DTranspose**

---

## U

### Upsampling
Proceso de aumentar la resolución espacial de un tensor.

**Métodos:**
- `UpSampling2D`: Duplica píxeles (no determinista en GPU)
- `Conv2DTranspose`: Convolución inversa (determinista, usado en proyecto)

---

## V

### Validación Cruzada (Cross-Validation)
Técnica que evalúa modelos en múltiples particiones del dataset para estimar performance generalizada.

**Validación del proyecto:** Split simple (70/15/15) en lugar de k-fold por limitación temporal de datos.

### Variable Regionalizada
Variable distribuida en el espacio con estructura de correlación espacial.

**Ejemplo:** Campo de precipitación Z(s) donde s es ubicación geográfica.

### Varianza
Medida de dispersión: promedio de desviaciones al cuadrado respecto a la media.

**Varianza ERA5 (pre-normalización):** 34.40 mm²/día²

### Variograma
Función que describe cómo la varianza entre pares de puntos aumenta con la distancia.

**Modelo ajustado:** Esférico con range=8.15°, sill=23.67, nugget=0.

**Aplicaciones:**
1. Diseño de receptive field CNN
2. Kriging para interpolación
3. Pesos espaciales en loss function

### Variograma Experimental
Estimación empírica del variograma a partir de datos observados antes de ajustar modelo teórico.

**Cálculo:**
```
γ̂(h) = (1/2|N(h)|) * Σ[Z(sᵢ) - Z(sⱼ)]²
```

### Variograma Teórico
Modelo paramétrico (esférico, exponencial, gaussiano) ajustado al variograma experimental.

**Ventaja:** Suaviza ruido, permite interpolación, garantiza propiedades matemáticas.

---

## W

### Weight Decay → Ver **L2 Regularization**

### Weighted Loss
Función de pérdida donde diferentes muestras o píxeles tienen ponderaciones distintas.

**Implementación:**
```python
weighted_error = squared_error * spatial_weights
```

### Window Size
Tamaño de la ventana temporal de observaciones pasadas usadas para predicción.

**Valor usado:** 7 días (una semana).

---

## 📊 SIGLAS Y ACRÓNIMOS

| Sigla | Significado | Contexto |
|-------|-------------|----------|
| **AE** | Autoencoder | Arquitectura de red neuronal |
| **API** | Application Programming Interface | Interfaz de programación |
| **BN** | Batch Normalization | Técnica de normalización |
| **CHIRPS** | Climate Hazards Infrared Precipitation with Stations | Dataset satelital |
| **CNN** | Convolutional Neural Network | Red neuronal convolucional |
| **CSV** | Comma-Separated Values | Formato de archivo |
| **DL** | Deep Learning | Aprendizaje profundo |
| **DMD** | Dynamic Mode Decomposition | Descomposición de modos dinámicos |
| **DRY** | Don't Repeat Yourself | Principio de programación |
| **ECMWF** | European Centre for Medium-Range Weather Forecasts | Centro meteorológico europeo |
| **EDA** | Exploratory Data Analysis | Análisis exploratorio |
| **ERA5** | ECMWF Reanalysis v5 | Dataset de reanálisis |
| **GFS** | Global Forecast System | Sistema de pronóstico NOAA |
| **GPU** | Graphics Processing Unit | Procesador gráfico |
| **KoVAE** | Kolmogorov-Arnold Variational Autoencoder | Variante de autoencoder |
| **LSTM** | Long Short-Term Memory | Arquitectura de RNN |
| **MAE** | Mean Absolute Error | Error absoluto medio |
| **MCP** | Model Context Protocol | Protocolo de contexto |
| **MSE** | Mean Squared Error | Error cuadrático medio |
| **NetCDF** | Network Common Data Form | Formato de datos científicos |
| **OK** | Ordinary Kriging | Kriging ordinario |
| **PCA** | Principal Component Analysis | Análisis de componentes principales |
| **PDO** | Pacific Decadal Oscillation | Oscilación decenal del Pacífico |
| **ReLU** | Rectified Linear Unit | Unidad lineal rectificada |
| **RF** | Receptive Field | Campo receptivo |
| **RMSE** | Root Mean Squared Error | Raíz del error cuadrático medio |
| **RNN** | Recurrent Neural Network | Red neuronal recurrente |
| **SAM** | Southern Annular Mode | Modo anular del sur |
| **SEED** | Semilla | Valor inicial aleatorio |
| **SVD** | Singular Value Decomposition | Descomposición en valores singulares |
| **TF** | TensorFlow | Framework de deep learning |
| **UDD** | Universidad del Desarrollo | Institución académica |
| **UMAP** | Uniform Manifold Approximation and Projection | Técnica de reducción dimensional |
| **VAE** | Variational Autoencoder | Autoencoder variacional |
| **WRF** | Weather Research and Forecasting | Modelo meteorológico |

---

## 🧮 FÓRMULAS PRINCIPALES

### 1. Variograma Esférico
```
γ(h) = {
    nugget + sill * [1.5(h/range) - 0.5(h/range)³]  si h ≤ range
    nugget + sill                                    si h > range
}
```

**Parámetros del proyecto:**
- nugget = 0.0
- sill = 23.67 mm²/día²
- range = 8.15° ≈ 905 km

---

### 2. Kriging Ordinario

**Sistema de ecuaciones:**
```
┌                    ┐   ┌    ┐   ┌       ┐
│ γ(s₁,s₁) ... γ(s₁,sₙ) │ │ 1 │   │ λ₁    │   │ γ(s₀,s₁) │
│    ⋮      ⋱      ⋮    │ │ ⋮ │ × │  ⋮    │ = │    ⋮     │
│ γ(sₙ,s₁) ... γ(sₙ,sₙ) │ │ 1 │   │ λₙ    │   │ γ(s₀,sₙ) │
│    1     ...    1     │ │ 0 │   │ μ     │   │    1     │
└                    ┘   └    ┘   └       ┘   └         ┘
```

**Predictor:**
```
Z*(s₀) = Σᵢ₌₁ⁿ λᵢ Z(sᵢ)
```

**Varianza kriging:**
```
σ²ₖ(s₀) = Σᵢ₌₁ⁿ λᵢ γ(s₀,sᵢ) + μ
```

---

### 3. Weighted MSE Loss
```
L_weighted = (1/N) Σᵢ₌₁ᴺ wᵢ * (yᵢ - ŷᵢ)²

donde wᵢ = 1 / (σ²ₖ(sᵢ) + ε)
```

**ε:** Pequeña constante para estabilidad numérica.

---

### 4. Batch Normalization
```
BN(x) = γ * ((x - μ_batch) / √(σ²_batch + ε)) + β

μ_batch = (1/m) Σᵢ₌₁ᵐ xᵢ
σ²_batch = (1/m) Σᵢ₌₁ᵐ (xᵢ - μ_batch)²
```

**γ, β:** Parámetros aprendibles.

---

### 5. Receptive Field (CNN)
Para convoluciones de tamaño k con dilation d:
```
RF_out = RF_in + (k-1) * d
```

**Ejemplo (dilation=[1,2,4,8], k=3):**
```
Capa 1: RF = 1 + 2*1 = 3
Capa 2: RF = 3 + 2*2 = 7
Capa 3: RF = 7 + 2*4 = 15
Capa 4: RF = 15 + 2*8 = 31
+ pooling: RF ≈ 40 celdas
```

---

### 6. Métricas de Evaluación

**MAE (Mean Absolute Error):**
```
MAE = (1/n) Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

**RMSE (Root Mean Squared Error):**
```
RMSE = √[(1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²]
```

**R² (Coeficiente de Determinación):**
```
R² = 1 - (SS_res / SS_tot)

SS_res = Σᵢ (yᵢ - ŷᵢ)²
SS_tot = Σᵢ (yᵢ - ȳ)²
```

---

### 7. StandardScaler (Normalización)
```
Transformación: z = (x - μ) / σ

μ = (1/n) Σᵢ₌₁ⁿ xᵢ
σ = √[(1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²]

Inversa: x = z * σ + μ
```

---

### 8. DMD (Dynamic Mode Decomposition)

**Descomposición SVD:**
```
X = U Σ Vᵀ
```

**Matriz de transición:**
```
A = X' V Σ⁻¹ Uᵀ
```

**Eigenvalues y modos:**
```
A Φ = Φ Λ
```

**Proyección temporal:**
```
x(t+h) = Aʰ x(t)
```

---

## 📖 REFERENCIAS BIBLIOGRÁFICAS

### Geoestadística
1. **Cressie, N. (1993).** Statistics for Spatial Data. Wiley.
2. **Chilès, J.-P., & Delfiner, P. (2012).** Geostatistics: Modeling Spatial Uncertainty. Wiley.
3. **Webster, R., & Oliver, M. A. (2007).** Geostatistics for Environmental Scientists. Wiley.

### Deep Learning
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** Deep Learning. MIT Press.
5. **Chollet, F. (2021).** Deep Learning with Python. Manning.

### DMD y Sistemas Dinámicos
6. **Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016).** Dynamic Mode Decomposition. SIAM.
7. **Schmid, P. J. (2010).** Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics.

### Meteorología y Climatología
8. **Hersbach, H., et al. (2020).** The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society.
9. **Funk, C., et al. (2015).** The climate hazards infrared precipitation with stations (CHIRPS) dataset. Scientific Data.

---

## 📝 NOTAS DE USO

### Convenciones en el Glosario

- **Negrita:** Término definido
- *Cursiva:* Énfasis o término técnico
- `Código`: Sintaxis de programación
- **→ Ver:** Referencia cruzada

### Sugerencias de Lectura

**Para principiantes:**
- Leer en orden: A → V → K → M → L
- Enfocarse en conceptos fundamentales antes de técnicas avanzadas

**Para usuarios avanzados:**
- Buscar términos específicos en el índice alfabético
- Revisar fórmulas matemáticas para implementación

**Para revisores del proyecto:**
- Sección "Siglas y Acrónimos" para decodificar documentación
- Sección "Fórmulas Principales" para validación matemática

---

## 🔄 CONTROL DE VERSIONES

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0 | 23-Nov-2025 | Versión inicial completa |

---

**Autor:** César Godoy Delaigue  
**Proyecto:** Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile  
**Institución:** Universidad del Desarrollo (UDD)  
**Contacto:** [tu_email@udd.cl]

---

_Este glosario es un documento vivo que se actualizará conforme avance el proyecto._
