# Optimización del Viajante (TSP Solver)

Este programa es una implementación de un solucionador para el **Problema del Viajante de Comercio (TSP)** utilizando varias heurísticas y algoritmos. La interfaz gráfica de usuario está basada en **Tkinter**, y permite al usuario generar puntos de entrega aleatorios, aplicar diferentes algoritmos para resolver el TSP y visualizar las soluciones en un mapa de fondo.

## Características

- **Generación de Puntos de Entrega**: Genera de manera aleatoria 20 puntos sobre el mapa y los visualiza.
- **Algoritmos de Optimización**:
  - **Vecino Más Cercano (Nearest Neighbor)**: Heurística simple que selecciona la ciudad más cercana en cada paso.
  - **Búsqueda Local (2-opt)**: Método que mejora una solución inicial mediante pequeñas modificaciones.
  - **Algoritmo Genético**: Optimización usando técnicas evolutivas, como selección, cruce y mutación.
  - **Árbol de Expansión Mínima (MST)**: Aproximación utilizando el algoritmo de mínimo coste para generar una ruta inicial.
- **Animación de la Solución**: Visualiza el recorrido óptimo en un gráfico interactivo.

## Requisitos

Para ejecutar este programa, necesitarás las siguientes librerías de Python:

- `tkinter` (incluido en la mayoría de las instalaciones de Python)
- `networkx` (para la manipulación de grafos)
- `matplotlib` (para la visualización gráfica y animación)
- `deap` (para implementar el algoritmo genético)
- `Pillow` (para manejo de imágenes)
- `numpy` (para operaciones numéricas)

Puedes instalar las dependencias con el siguiente comando:

```bash
pip install networkx matplotlib deap Pillow numpy
```

## Instrucciones de Uso

1. **Ejecuta el programa**:
   - Al ejecutar el script, se abrirá una ventana con el mapa de fondo y un conjunto de botones.
  
2. **Genera los puntos de entrega**:
   - Haz clic en el botón **"Generar Puntos de entrega"** para generar aleatoriamente 20 puntos sobre el mapa. Estos puntos representarán las ciudades que el viajante debe visitar.

3. **Selecciona un algoritmo**:
   - Puedes elegir entre varios algoritmos para resolver el TSP:
     - **Vecino Más Cercano**
     - **Búsqueda Local**
     - **Algoritmo Genético**
     - **Árbol de Expansión Mínima**

4. **Visualización**:
   - Después de elegir un algoritmo, la ruta óptima o aproximada será visualizada en el mapa, donde las ciudades serán conectadas con líneas rojas.

## Funciones Principales

- **Generación de Ciudades**:  
  La función `generate_cities` genera 20 puntos aleatorios dentro de los límites del mapa.
  
- **Resolución del TSP**:
  - **Vecino Más Cercano**: Implementado en `solve_tsp`.
  - **Búsqueda Local**: Implementado en `solve_local_search` utilizando el algoritmo 2-opt.
  - **Algoritmo Genético**: Implementado en `solve_genetic` usando la librería `deap`.
  - **Árbol de Expansión Mínima (MST)**: Implementado en `solve_mst` usando `networkx`.
  - **Held-Karp**: Implementado en `solve_held_karp` para soluciones exactas (limitadas a 20 ciudades).

- **Animación de la Solución**:  
  La función `animate_solution` se encarga de visualizar las rutas generadas mediante animaciones en **matplotlib**.

## Consideraciones

- La visualización puede requerir ajustes dependiendo del tamaño del mapa de fondo utilizado.

---

## Licencia

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

You are free to:

Share — copy and redistribute the material in any medium or format.

Adapt — remix, transform, and build upon the material.

Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

NonCommercial — You may not use the material for commercial purposes.

ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

Read more about this license at CC BY-NC-SA 4.0.
