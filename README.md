# Projet : Déployez un modèle dans le cloud

**Autor :** Franck Le mat

**Date :** 10/05/2022

**Durée totale :** 70 heures


## Background du projet :
Vous êtes Data Scientist dans une très jeune start-up de l'AgriTech, nommée  "Fruits!", qui cherche à proposer des solutions innovantes pour la récolte des fruits.
La volonté de l’entreprise est de préserver la biodiversité des fruits en permettant des traitements spécifiques pour chaque espèce de fruits en développant des robots cueilleurs intelligents.
Votre start-up souhaite dans un premier temps se faire connaître en mettant à disposition du grand public une application mobile qui permettrait aux utilisateurs de prendre en photo un fruit et d'obtenir des informations sur ce fruit.
Pour la start-up, cette application permettrait de sensibiliser le grand public à la biodiversité des fruits et de mettre en place une première version du moteur de classification des images de fruits.
De plus, le développement de l’application mobile permettra de construire une première version de l'architecture Big Data nécessaire.
Nous cherchons à anticiper le fait que le volume de données va augmenter très rapidement, ainsi nous travaillerons via des outils Big Data pour se préparer au passage à l'échelle. Enfin, nous déploierons ce modèle dans le cloud (Amazon Web Services).

Données : https://www.kaggle.com/moltean/fruits

## Contraintes :
Lors de son brief initial, Paul vous a averti des points suivants :

Vous devrez tenir compte dans vos développements du fait que le volume de données va augmenter très rapidement après la livraison de ce projet. Vous développerez donc des scripts en Pyspark et utiliserez par exemple le cloud AWS pour profiter d’une architecture Big Data (EC2, S3, IAM), basée sur un serveur EC2 Linux.
La mise en œuvre d’une architecture Big Data sous (par exemple) AWS peut nécessiter une configuration serveur plus puissante que celle proposée gratuitement (EC2 = t2.micro, 1 Go RAM, 8 Go disque serveur).

## Key point du projet :

- Utiliation des services S3 pour stockage des données
- Configuration d'une instance AWS EC2 de taille t2.xlarge (OS Ubuntu Server 18.04)
- Réalisation de scripts pyspark Exécution dans le cloud :
    - Computer Vision : Transfer Learning & Feature Extraction (VGG-16) via la librairie Keras
- Lecture et enregistrement de données sur Amazon S3


## Livrables :
- Un notebook sur le cloud contenant les scripts en Pyspark exécutables (le preprocessing et une étape de réduction de dimension).
- Les images du jeu de données initial ainsi que la sortie de la sortie de la réduction de dimension (une matrice écrite sur un fichier CSV ou autre) disponible dans un espace de stockage sur le cloud.
- Un support de présentation pour la soutenance, présentant :
      - les différentes briques d'architecture choisies sur le cloud ;
      - leur rôle dans l’architecture Big Data ;
      - les étapes de la chaîne de traitement.


## Compétences évaluées :
- Paralléliser des opérations de calcul avec Pyspark
- Utiliser les outils du cloud pour manipuler des données dans un environnement Big Data
- Identifier les outils du cloud permettant de mettre en place un environnement Big Data

