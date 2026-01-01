# Dashboard Microfinance - Analyse de Risque Crédit

Tableau de bord interactif développé avec Dash pour l'analyse du portefeuille de crédit et la prédiction du risque de défaut dans le secteur de la microfinance.

## Description du Projet

Cette application web permet aux institutions de microfinance d'analyser leur portefeuille de clients, d'identifier les facteurs de risque et de prédire les probabilités de défaut à l'aide de modèles statistiques avancés. Le dashboard offre des visualisations interactives, des outils de filtrage dynamiques et des capacités de modélisation prédictive.

## Structure de la Base de Données

Le fichier de données `microfinance_credit_risk.xlsx` contient les informations suivantes pour chaque client :

### Identifiant et Localisation
- **client_id** : Identifiant unique du client
- **region** : Région géographique du client (Dakar, Thiès, Saint-Louis, Kaolack, Ziguinchor)

### Informations Démographiques
- **age** : Âge du client en années
- **sexe** : Genre du client (Homme, Femme)
- **nb_dependants** : Nombre de personnes à charge

### Informations Financières
- **revenu_mensuel_xof** : Revenu mensuel en francs CFA
- **epargne_xof** : Montant d'épargne disponible en francs CFA
- **anciennete_emploi_mois** : Ancienneté dans l'emploi actuel en mois

### Caractéristiques du Prêt
- **montant_pret_xof** : Montant du prêt accordé en francs CFA
- **duree_mois** : Durée du prêt en mois
- **taux_interet_annuel_pct** : Taux d'intérêt annuel en pourcentage
- **dsti_pct** : Ratio dette/revenu (Debt Service to Income) en pourcentage

### Informations Commerciales
- **secteur_activite** : Secteur d'activité du client (Commerce, Agriculture, Services, Artisanat, Transport)
- **canal_octroi** : Canal de distribution du prêt (Agence, Mobile Money, Agent)
- **type_garantie** : Type de garantie fournie (Caution solidaire, Hypothèque, Gage, Sans garantie)
- **pret_groupe** : Indique si le prêt est individuel ou de groupe (0 = Individuel, 1 = Groupe)

### Historique et Comportement
- **historique_credit** : Nombre de prêts antérieurs
- **jours_retard_12m** : Nombre de jours de retard cumulés sur les 12 derniers mois
- **utilise_mobile_money** : Utilisation des services de mobile banking (0 = Non, 1 = Oui)

### Variable Cible
- **defaut_90j** : Statut de défaut à 90 jours (0 = Pas de défaut, 1 = Défaut)

## Structure du Projet

```
Dashboard-Microfinance/
│
├── app.py                          # Application principale Dash
├── callbacks.py                    # Logique interactive des callbacks
│
├── page_home.py                    # Page d'accueil
├── page_analysis.py                # Page d'analyse et exploration
├── page_modeling.py                # Page de modélisation prédictive
│
├── assets/                         # Ressources statiques
│   ├── logo.png                    # Logo de l'institution
│   ├── navbar_styles.css           # Styles de la barre de navigation
│   ├── home_styles.css             # Styles de la page d'accueil
│   └── analysis_styles.css         # Styles de la page d'analyse
│
└── Data/
    └── microfinance_credit_risk.xlsx                    # Base de données clients
```

## Installation et Exécution

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des Dépendances

```bash
pip install dash dash-bootstrap-components pandas plotly numpy scikit-learn
```

### Lancement de l'Application

```bash
python app.py
```

L'application sera accessible à l'adresse suivante : `http://127.0.0.1:8050`

## Fonctionnalités Principales

### Page d'Accueil
La page d'accueil présente une vue d'ensemble du portefeuille avec un carousel d'images illustrant les activités de microfinance, des cartes de navigation vers les différentes sections analytiques et un aperçu des domaines d'intervention institutionnels.

### Page Analyse et Exploration
Cette section offre des outils d'analyse approfondie incluant des filtres interactifs par région et statut de défaut, quatre indicateurs clés de performance mis à jour dynamiquement, une carte choroplèthe interactive du Sénégal visualisant les taux de défaut par région, des graphiques de performance comparant objectifs et réalisations, ainsi qu'un tableau détaillé avec mise en évidence des clients en défaut.

### Page Modélisation
La page de modélisation permet de comparer les performances de deux algorithmes discriminants linéaire et quadratique, d'analyser les métriques de performance incluant précision, rappel et score F1, de visualiser les matrices de confusion et les courbes ROC, et d'utiliser un simulateur de prédiction en temps réel pour évaluer le risque de défaut de nouveaux clients.

## Technologies Utilisées

- **Dash** : Framework web pour applications analytiques interactives
- **Plotly** : Bibliothèque de visualisation interactive
- **Pandas** : Manipulation et analyse de données
- **Scikit-learn** : Modélisation prédictive et apprentissage automatique
- **Dash Bootstrap Components** : Composants UI stylisés

## Charte Graphique

Le dashboard utilise une palette de couleurs institutionnelle inspirée des couleurs naturelles africaines avec un vert foncé représentant la croissance et la prospérité, un marron foncé évoquant la terre et la stabilité, un navy foncé symbolisant la confiance et le professionnalisme, et des accents dorés apportant une touche d'élégance et de valeur.

## Auteur

Projet développé dans le cadre de l'analyse de risque crédit pour les institutions de microfinance.

## Licence

Ce projet est destiné à un usage éducatif et professionnel dans le domaine de la microfinance.
