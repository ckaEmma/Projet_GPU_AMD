# PPN : Programmation GPU AMD, benchmark et analyse de performances 

Projet de recherche encadré portant sur l'étude de la programmation HPC sur GPU AMD, le développement de benchmarks synthétiques et l'évaluation critique des outils de profiling.

## Structure du dépôt

* `src/` : Code source des benchmarks et MiniApps (HIP, OpenMP, rocBLAS, etc.).
* `scripts/` : Scripts d'automatisation pour la compilation, l'exécution et le profilage.
* `docs/` : Documentation technique et cahier de laboratoire (journal de bord).
* `report/` : Rapport final et supports de présentation.

## Environnement & outils

Pour compiler et exécuter le code, l'environnement **AMD ROCm** est requis.
* **Version de ROCm :** ROCm 7.1.1
* **Compilateur :** `hipcc` / `g++`
* **Bibliothèques requises :** 
  * rocBLAS
  * OpenMP
  
## Auteurs
* Mehdi ABDENNOUR
* Emma CHAKMA
* Kocouvi Klaus KOUDRI
* Fredy PONTURAT