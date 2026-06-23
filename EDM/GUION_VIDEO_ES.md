# 🎬 Guion del vídeo — *Sensorless Traffic* (objetivo ≤ 5:00)

**Formato:** grabación de pantalla de la app + voz en off.
**Equipo:** Imad, Raúl y Nouh — repartid la narración (una sección cada uno) para
que se vea como un vídeo de equipo; usad "nosotros / nuestro equipo".
**Consejo:** ensayad una vez; los tiempos dejan ~15 s de margen. Hablad con calma
— es mejor terminar en 4:50 que ir acelerados.

> ⚠️ **Recordatorio:** el enunciado pide que el vídeo sea **en inglés**. Este guion
> en español es para que entendáis y ensayéis; tenéis la versión en inglés en
> `VIDEO_SCRIPT.md` (y su PDF) para leer en cámara.

---

### 0:00 – 0:35 · Gancho, equipo y el problema de la ciudad
> "Imaginad un departamento de tráfico de una ciudad decidiendo dónde añadir un
> carril bus o calmar el tráfico de una calle. Para hacerlo bien, necesitan saber
> cuánto tráfico tiene cada calle. ¿El problema? Una ciudad solo *conoce* el
> tráfico de las calles donde instaló un sensor — y en Valencia eso es apenas el
> **2,4 %** de ellas. El otro **97,6 % son puntos ciegos**, y un pueblo cercano
> como Paiporta **no tiene ningún sensor**."
>
> "Somos **Imad, Raúl y Nouh**, y nuestra app **Sensorless Traffic** convierte esa
> pequeña fracción medida en un mapa de congestión de *todas* las calles — incluso
> en pueblos que nunca han tenido un solo sensor."

🖥️ *Abrid la app en la pestaña del mapa, con las métricas principales visibles.
Dejad que el mapa termine de cargar antes de hablar sobre él.*

---

### 0:35 – 1:25 · Pestaña 1 — el problema de los puntos ciegos, visible
> "Esta es toda la red viaria de Valencia — unos 54.000 tramos de calle. Por
> defecto mostramos la **estimación de congestión de nuestro modelo para toda la
> ciudad**: verde es fluido, rojo es demanda de arteria principal."

🖥️ *Pasad el ratón por un par de calles para mostrar el tooltip (nombre + nivel).
Señalad la leyenda en el mapa, abajo a la derecha.*

> "Ahora mirad lo que la ciudad *realmente* mide."

🖥️ *Cambiad la capa a "Sensor coverage only". El mapa se vuelve casi todo gris.*

> "Todo lo gris es un punto ciego — solo estas pocas calles de color tienen un
> sensor. Nuestro objetivo es rellenar todo el gris, de forma fiable."

🖥️ *Volved a "Model estimate".*

---

### 1:25 – 2:35 · Pestaña 4 — metodología y cómo de bien funciona
> "¿Cómo? Con un pipeline completo de ciencia de datos — el proceso CRISP-DM de la
> asignatura. Descargamos la red viaria de **OpenStreetMap** y los conteos por hora
> de los **sensores de espira de Valencia**, y limpiamos e imputamos los datos
> faltantes: la mediana de carriles por tipo de vía, y un modelo de **k-vecinos más
> cercanos espacial** para los límites de velocidad que faltan. Un algoritmo propio
> de Haversine *punto-sobre-tramo* asigna cada sensor a su calle."

🖥️ *Recorred la fila de metodología de 4 pasos.*

> "Agrupamos el tráfico en cuatro niveles de congestión, confirmamos con tests de
> **ANOVA y Tukey** que el tipo de vía y la velocidad los separan de verdad, y
> comparamos cinco modelos — desde una línea base hasta un **árbol de decisión, un
> Random Forest y finalmente XGBoost**, que gana. Esa es la historia de
> bagging-contra-boosting del Tema 2."

🖥️ *Señalad las cinco tarjetas de métricas, luego el gráfico de importancia de variables.*

> "Y lo evaluamos con honestidad, con las métricas del Tema 1: **F1-macro 0,53**,
> **kappa de Cohen 0,39**, un **AUC macro de 0,77** — unas **5 veces una línea base**,
> usando *solo la forma estática de la calle*. La matriz de confusión muestra que
> los errores caen entre niveles *adyacentes*, las curvas ROC muestran que la
> congestión pico es la más separable, y la **curva de calibración** — con un error
> de calibración esperado de solo **0,04** — significa que la confianza del modelo
> es fiable."

🖥️ *Recorred matriz de confusión → curvas ROC → gráfico de calibración. Luego abrid
brevemente el desplegable "Live data feed" para mostrar el feed en vivo 🟢 de Valencia.*

---

### 2:35 – 3:30 · Pestaña 2 — el simulador de escenarios (el beneficio)
> "Aquí es donde se convierte en una herramienta de planificación. Elige cualquier
> calle…"

🖥️ *Seleccionad una calle reconocible del desplegable; mostrad su nivel + las barras
de probabilidad.*

> "…y pregunta *¿y si la rediseñáramos?* Es **inferencia en vivo con el mismo modelo
> XGBoost — no reglas programadas a mano**. Vamos a añadir carriles y subir el límite
> de velocidad."

🖥️ *Subid el slider de carriles, subid el límite de velocidad; las barras se desplazan
hacia alto/pico en vivo.*

> "El modelo reestima la congestión al instante. Un planificador puede valorar ese
> compromiso **antes** de gastar un solo euro — y sin instalar un sensor."

---

### 3:30 – 4:20 · Pestaña 3 — transferencia a un pueblo sin sensores
> "Por último, la parte de la que estamos más orgullosos. Paiporta, junto a Valencia,
> tiene **cero** sensores de tráfico. Cogemos el modelo entrenado solo con Valencia,
> más los priors por tipo de vía que aprendió, y producimos un **mapa de congestión
> completo para Paiporta** — enteramente a partir de la geometría de OpenStreetMap."

🖥️ *Abrid la pestaña de transferencia; mostrad el mapa de Paiporta y las métricas de
"0 sensores / 100 % de cobertura".*

> "Cada calle de color aquí se estimó sin ninguna verdad de terreno local. Ese es el
> verdadero valor: un método que le da a *cualquier* pueblo su primer mapa de
> congestión gratis."

---

### 4:20 – 4:55 · Cierre
> "Así que — datos abiertos más un pipeline de ML transparente convierten el 2,4 % de
> calles medidas en un mapa de congestión de toda la ciudad, *y* de todo el pueblo,
> con un simulador what-if en vivo para planificadores. Está desplegado como una app
> server-side que combina predicciones en batch con un feed en tiempo real — justo
> los patrones de despliegue del Tema 5. El código, la app en vivo y las fuentes de
> datos están todos en la descripción. Gracias por ver el vídeo."

🖥️ *Terminad con la URL de la app en vivo / el repo de GitHub en pantalla.*

---

## Checklist antes de grabar
- [ ] App **desplegada** y abierta en una ventana limpia del navegador (ocultad la barra de marcadores; pantalla completa).
- [ ] La **pestaña en vivo** muestra el feed en tiempo real 🟢 (funciona desde una red normal).
- [ ] En la pestaña 1, el **mapa ha terminado de cargar** antes de empezar a hablar sobre él.
- [ ] Elegid una **calle reconocible** para el simulador (p. ej. una avenida conocida) para que el público conecte.
- [ ] Tened la **pestaña del modelo pre-desplazada** para que ROC + calibración estén a un solo paneo.
- [ ] Mantened el total **por debajo de 5:00** — la rúbrica es estricta con la duración. Haced un ensayo cronometrado.
- [ ] Repartid la narración entre **Imad, Raúl y Nouh** (una sección cada uno) para cumplir lo de "vídeo del equipo".

## Frases sueltas por si os sobran segundos
- *Originalidad:* "Predecimos la congestión **donde no hay sensor** — y la transferimos a un pueblo que no tiene ninguno."
- *Dificultad:* "Grafos de OSM, imputación kNN espacial, un algoritmo propio de emparejamiento geográfico, cinco modelos comparados y un despliegue en vivo."
- *Métodos de DS:* "Imputación, ANOVA/Tukey, boosting, validación cruzada, ROC/AUC y calibración."
