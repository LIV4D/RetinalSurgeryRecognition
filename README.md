# Retinal Surgery Phases Recognition
Project: Phases recognition in retinal surgery

## Structure du répertoire

Ce répertoire est composé du dossier doc/ qui contiendra toute éventuelle documentation, et src/, qui contiendra ton code.

Il contiendra également certainement un dossier data/ et un dossier exp/

Le dossier data/ pour les données d'entrées et exp/ pour chacunes des expériences qui seront réalisées.

Ces dossiers ne seront pas trackés par git (car beaucoup trop lourd pour l'hébergement sur github), ils resteront donc en local là où les expériences rouleront.

## Squelette de code
test
Dans src/, j'ai indiqué un squelette de code possible pour l'entrainement d'un réseau de type CNN. Il y a tout un tas de fonctions à compléter en fonction des besoins, rien n'est figé dans le marbre.

Le point d'entrée du programme est le fichier main.py. Il construit un Trainer, qui est une classe dérivée de AbstractManager. L'idée est de créer autant de Trainer que d'expériences qu'on souhaite mener. Potentiellement, il sera peut être nécessaire d'ajouter un Tester, qui remplit le même rôle que le Trainer, mais pour l'évaluation d'un modèle entraîné