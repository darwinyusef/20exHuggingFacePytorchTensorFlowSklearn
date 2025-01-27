# Plan de Estudio de 8 Semanas para Aprender Scikit-learn y Machine Learning

## **Semana 1: Fundamentos de ML y Primeros Pasos con Sklearn**
**Objetivo:** Entender conceptos clave de ML y familiarizarte con la API de Sklearn.

1. **Conceptos teóricos:**
   - Tipos de ML: Supervisado (clasificación/regresión) vs. No Supervisado (clustering).
   - Flujo de trabajo en ML: `datos → preprocesamiento → entrenamiento → evaluación → predicción`.
   - Overfitting vs. Underfitting.

2. **Práctica con Sklearn:**
   - Instalación: `pip install scikit-learn`.
   - Cargar datasets de ejemplo: `load_iris()`, `load_digits()`, `load_boston()`.
   - Usar `train_test_split` para dividir datos.
   - Entrenar y evaluar un modelo simple (ej: `KNeighborsClassifier` o `LinearRegression`).

3. **Recurso clave:** [Tutorial oficial de Sklearn](https://scikit-learn.org/stable/tutorial/basic/tutorial.html).

---

## **Semana 2-3: Aprendizaje Supervisado (Clasificación)**
**Objetivo:** Dominar algoritmos de clasificación y métricas de evaluación.

1. **Algoritmos:**
   - Regresión Logística (`LogisticRegression`).
   - Árboles de Decisión (`DecisionTreeClassifier`).
   - Random Forest (`RandomForestClassifier`).
   - SVM (`SVC`).

2. **Métricas:**
   - Precisión, Recall, F1-Score (`classification_report`).
   - Matriz de Confusión (`confusion_matrix`).

3. **Proyecto práctico:**
   - Dataset: [Titanic en Kaggle](https://www.kaggle.com/c/titanic) (predecir supervivencia).
   - **Recurso:** [Guía de clasificación con Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

---

## **Semana 4: Aprendizaje Supervisado (Regresión)**
**Objetivo:** Resolver problemas de regresión y entender métricas clave.

1. **Algoritmos:**
   - Regresión Lineal (`LinearRegression`).
   - Regresión Polinómica (`PolynomialFeatures` + `LinearRegression`).
   - Random Forest para regresión (`RandomForestRegressor`).

2. **Métricas:**
   - MSE, RMSE, R² (`mean_squared_error`, `r2_score`).

3. **Proyecto práctico:**
   - Dataset: [Boston Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) (predecir precios de viviendas).
   - **Recurso:** [Guía de regresión con Sklearn](https://scikit-learn.org/stable/modules/linear_model.html).

---

## **Semana 5: Preprocesamiento de Datos**
**Objetivo:** Aprender a limpiar y preparar datos para modelos.

1. **Herramientas clave:**
   - Escalado: `StandardScaler`, `MinMaxScaler`.
   - Codificación de variables categóricas: `OneHotEncoder`, `LabelEncoder`.
   - Manejo de valores faltantes: `SimpleImputer`.
   - Pipelines: `Pipeline` y `ColumnTransformer`.

2. **Práctica:**
   - Preprocesar el dataset [Titanic](https://www.kaggle.com/c/titanic) usando pipelines.
   - **Recurso:** [Guía de preprocesamiento](https://scikit-learn.org/stable/modules/preprocessing.html).

---

## **Semana 6-7: Optimización de Modelos**
**Objetivo:** Mejorar el rendimiento de tus modelos con técnicas avanzadas.

1. **Validación cruzada:**
   - `cross_val_score` y `KFold`.

2. **Ajuste de hiperparámetros:**
   - Grid Search (`GridSearchCV`).
   - Random Search (`RandomizedSearchCV`).

3. **Selección de características:**
   - `SelectKBest`, `RFE` (Eliminación Recursiva de Características).

4. **Práctica:**
   - Optimizar un modelo de Random Forest para el dataset Titanic.
   - **Recurso:** [Guía de optimización](https://scikit-learn.org/stable/modules/grid_search.html).

---

## **Semana 8: Aprendizaje No Supervisado y Proyecto Final**
**Objetivo:** Explorar clustering y reducción de dimensionalidad.

1. **Algoritmos:**
   - Clustering: K-Means (`KMeans`), DBSCAN (`DBSCAN`).
   - Reducción de dimensionalidad: PCA (`PCA`), t-SNE (`TSNE`).

2. **Proyecto integrador:**
   - Dataset: [Mall Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) (agrupar clientes).
   - Combina todo lo aprendido: preprocesamiento, clustering y visualización.
   - **Recurso:** [Guía de clustering](https://scikit-learn.org/stable/modules/clustering.html).

---

## **Recursos Adicionales Recomendados**
1. **Libros:**
   - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Aurélien Géron).
   - "Python Data Science Handbook" (Jake VanderPlas).

2. **Cursos Gratuitos:**
   - [Kaggle Learn](https://www.kaggle.com/learn): Cursos prácticos de ML y Sklearn.
   - [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning) (teoría sólida).

3. **Comunidades:**
   - Foros: Stack Overflow, Reddit (r/learnmachinelearning).
   - Competencias: [Kaggle](https://www.kaggle.com/).

---

## **Consejos Clave**
1. **Aprende haciendo:** Dedica el 80% del tiempo a codificar y el 20% a teoría.
2. **Experimenta:** Modifica hiperparámetros, prueba distintos algoritmos y compara resultados.
3. **Documenta tu código:** Usa Jupyter Notebook para explicar cada paso y visualizar datos.
4. **No te atores:** Si un concepto no queda claro, sigue practicando y regresa después.

---

¡Manos a la obra! En 8 semanas tendrás un dominio sólido de Sklearn y podrás abordar proyectos reales de ML. Si te bloqueas o necesitas ejemplos específicos, ¡avísame! 😊
