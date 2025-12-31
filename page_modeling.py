import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, roc_curve, auc

# Palette de couleurs personnalisée
COULEURS = {
    'vert_fonce': '#2d5016',     
    'marron_fonce': '#4a3425',   
    'marron_secondaire': '#3d2a1f',  
    'navy_fonce': '#1a3a52',      
    'dore': '#c9a85c',            
    'blanc': '#ffffff',
    'gris_clair': "#fefeff",
    'texte_principal': '#2c3e50',
    'texte_secondaire': '#7f8c8d'
}

# ----------------------------------------------------------------------------
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ----------------------------------------------------------------------------

# Importation du fichier de données client de microfinance
donnees_clients = pd.read_excel('Data/microfinance_credit_risk.xlsx')


# ----------------------------------------------------------------------------
# SÉLECTION DES VARIABLES PRÉDICTIVES
# ----------------------------------------------------------------------------
# Choix rigoureux des features pertinentes pour la prédiction de défaut
# Critères de sélection : 
# - Variables quantifiables et objectives
# - Disponibles au moment de l'octroi du prêt
# - Corrélées avec le risque de défaut selon la littérature

variables_predictives = [
    'age',                          # Âge du client (années)
    'revenu_mensuel_xof',           # Revenu mensuel en francs CFA
    'epargne_xof',                  # Montant d'épargne en francs CFA
    'anciennete_relation_mois',     # Ancienneté de la relation bancaire (mois)
    'historique_credit_mois',       # Historique de crédit (mois)
    'jours_retard_12m',             # Jours de retard sur les 12 derniers mois
    'nb_dependants',                # Nombre de personnes à charge
    'usage_mobile_money_score',     # Score d'utilisation du mobile money (0-100)
    'montant_pret_xof',             # Montant du prêt demandé en francs CFA
    'duree_mois',                   # Durée du prêt en mois
    'taux_interet_annuel_pct',      # Taux d'intérêt annuel (%)
    'dsti_pct',                     # Debt Service to Income ratio (%)
    'pret_groupe'                   # Indicateur de prêt de groupe (0/1)
]

# Variable cible : défaut de paiement à 90 jours
variable_cible = 'defaut_90j'

# Extraction des matrices de features (X) et de la cible (y)
matrice_features = donnees_clients[variables_predictives].copy()
vecteur_cible = donnees_clients[variable_cible].copy()

# Gestion des valeurs manquantes par imputation à la moyenne
# Cette approche simple est acceptable car les valeurs manquantes sont rares
matrice_features = matrice_features.fillna(matrice_features.mean())


# ----------------------------------------------------------------------------
# DIVISION TRAIN/TEST AVEC STRATIFICATION
# ----------------------------------------------------------------------------
# Split 70/30 avec stratification pour maintenir la proportion de défauts
# dans les ensembles d'entraînement et de test
# Graine aléatoire fixée à 42 pour la reproductibilité

features_train, features_test, cible_train, cible_test = train_test_split(
    matrice_features,
    vecteur_cible,
    test_size=0.30,           # 30% des données pour le test
    random_state=42,          # Graine pour reproductibilité
    stratify=vecteur_cible    # Maintien de la proportion de défauts
)

# Dimensions des ensembles créés
nombre_exemples_train = len(features_train)
nombre_exemples_test = len(features_test)
taux_defaut_train = (cible_train.sum() / len(cible_train)) * 100
taux_defaut_test = (cible_test.sum() / len(cible_test)) * 100


# ----------------------------------------------------------------------------
# STANDARDISATION DES VARIABLES (Z-SCORE NORMALIZATION)
# ----------------------------------------------------------------------------
# La standardisation est essentielle pour LDA et QDA car ces algorithmes
# sont sensibles à l'échelle des variables. On centre et réduit chaque
# variable pour avoir moyenne=0 et écart-type=1

standardiseur = StandardScaler()

# Apprentissage des paramètres de standardisation sur l'ensemble train uniquement
# (pour éviter le data leakage)
features_train_standardisees = standardiseur.fit_transform(features_train)

# Application de la même transformation sur l'ensemble test
features_test_standardisees = standardiseur.transform(features_test)


# ----------------------------------------------------------------------------
# ENTRAÎNEMENT DU MODÈLE LDA (LINEAR DISCRIMINANT ANALYSIS)
# ----------------------------------------------------------------------------
# LDA suppose que :
# - Les classes ont la même matrice de covariance
# - Les données suivent une distribution normale multivariée
# - La frontière de décision est linéaire

modele_lda = LinearDiscriminantAnalysis()
modele_lda.fit(features_train_standardisees, cible_train)

# Prédictions sur l'ensemble test
predictions_lda = modele_lda.predict(features_test_standardisees)
probabilites_lda = modele_lda.predict_proba(features_test_standardisees)[:, 1]


# ----------------------------------------------------------------------------
# ENTRAÎNEMENT DU MODÈLE QDA (QUADRATIC DISCRIMINANT ANALYSIS)
# ----------------------------------------------------------------------------
# QDA suppose que :
# - Chaque classe a sa propre matrice de covariance
# - Les données suivent une distribution normale multivariée
# - La frontière de décision est quadratique (plus flexible que LDA)
# 
# Paramètre reg_param : régularisation pour gérer les matrices de covariance
# mal conditionnées (important quand nb_samples < nb_features)

modele_qda = QuadraticDiscriminantAnalysis(reg_param=0.4, solver ='eigen', shrinkage = 0.3)
modele_qda.fit(features_train_standardisees, cible_train)

# Prédictions sur l'ensemble test
predictions_qda = modele_qda.predict(features_test_standardisees)
probabilites_qda = modele_qda.predict_proba(features_test_standardisees)[:, 1]


# ----------------------------------------------------------------------------
# CALCUL DES MÉTRIQUES DE PERFORMANCE
# ----------------------------------------------------------------------------

# Métriques pour LDA
accuracy_lda = accuracy_score(cible_test, predictions_lda)
f1_score_lda = f1_score(cible_test, predictions_lda, zero_division=0)
recall_lda = recall_score(cible_test, predictions_lda, zero_division=0)
matrice_confusion_lda = confusion_matrix(cible_test, predictions_lda)

# Métriques pour QDA
accuracy_qda = accuracy_score(cible_test, predictions_qda)
f1_score_qda = f1_score(cible_test, predictions_qda, zero_division=0)
recall_qda = recall_score(cible_test, predictions_qda, zero_division=0)
matrice_confusion_qda = confusion_matrix(cible_test, predictions_qda)

# Calcul des courbes ROC
fpr_lda, tpr_lda, seuils_lda = roc_curve(cible_test, probabilites_lda)
auc_lda = auc(fpr_lda, tpr_lda)

fpr_qda, tpr_qda, seuils_qda = roc_curve(cible_test, probabilites_qda)
auc_qda = auc(fpr_qda, tpr_qda)

# Identification du meilleur modèle basé sur le F1-Score
# (métrique équilibrée entre précision et rappel)
if f1_score_lda >= f1_score_qda:
    nom_meilleur_modele = "LDA"
    f1_meilleur_modele = f1_score_lda
    modele_recommande = modele_lda
    couleur_meilleur = COULEURS['navy_fonce']
else:
    nom_meilleur_modele = "QDA"
    f1_meilleur_modele = f1_score_qda
    modele_recommande = modele_qda
    couleur_meilleur = COULEURS['marron_fonce']


# ----------------------------------------------------------------------------
# CRÉATION DE LA MATRICE DE CONFUSION - LDA
# ----------------------------------------------------------------------------

graphique_confusion_lda = go.Figure(data=go.Heatmap(
    z=matrice_confusion_lda,
    x=['Prédit: Non Défaut', 'Prédit: Défaut'],
    y=['Réel: Non Défaut', 'Réel: Défaut'],
    colorscale=[[0, COULEURS['blanc']], [1, COULEURS['navy_fonce']]],
    text=matrice_confusion_lda,
    texttemplate='%{text}',
    textfont={"size": 18, "color": 'lightblue'},
    showscale=False,
    hoverongaps=False
))

graphique_confusion_lda.update_layout(
    title={
        'text': 'Matrice de Confusion - LDA',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 14, 'color': COULEURS['texte_principal']}
    },
    xaxis_title='',
    yaxis_title='',
    height=320,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=11, color=COULEURS['texte_principal'])
)


# ----------------------------------------------------------------------------
# CRÉATION DE LA MATRICE DE CONFUSION - QDA
# ----------------------------------------------------------------------------

graphique_confusion_qda = go.Figure(data=go.Heatmap(
    z=matrice_confusion_qda,
    x=['Prédit: Non Défaut', 'Prédit: Défaut'],
    y=['Réel: Non Défaut', 'Réel: Défaut'],
    colorscale=[[0, COULEURS['blanc']], [1, COULEURS['marron_fonce']]],
    text=matrice_confusion_qda,
    texttemplate='%{text}',
    textfont={"size": 18, "color": "#f7e5be"},
    showscale=False,
    hoverongaps=False
))

graphique_confusion_qda.update_layout(
    title={
        'text': 'Matrice de Confusion - QDA',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 14, 'color': COULEURS['texte_principal']}
    },
    xaxis_title='',
    yaxis_title='',
    height=320,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=11, color=COULEURS['texte_principal'])
)


# ----------------------------------------------------------------------------
# CRÉATION DES COURBES ROC COMPARATIVES
# ----------------------------------------------------------------------------

graphique_courbes_roc = go.Figure()

# Courbe ROC pour LDA
graphique_courbes_roc.add_trace(go.Scatter(
    x=fpr_lda,
    y=tpr_lda,
    mode='lines',
    name=f'LDA (AUC = {auc_lda:.3f})',
    line=dict(color=COULEURS['navy_fonce'], width=3),
    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
))

# Courbe ROC pour QDA
graphique_courbes_roc.add_trace(go.Scatter(
    x=fpr_qda,
    y=tpr_qda,
    mode='lines',
    name=f'QDA (AUC = {auc_qda:.3f})',
    line=dict(color=COULEURS['marron_fonce'], width=3),
    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
))

# Ligne de référence (modèle aléatoire)
graphique_courbes_roc.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Aléatoire',
    line=dict(color=COULEURS['texte_secondaire'], width=2, dash='dash'),
    hovertemplate='Modèle aléatoire<extra></extra>'
))

graphique_courbes_roc.update_layout(
    title={
        'text': 'Courbes ROC Comparatives',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 14, 'color': COULEURS['texte_principal']}
    },
    xaxis_title='Taux de Faux Positifs (FPR)',
    yaxis_title='Taux de Vrais Positifs (TPR)',
    height=320,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color=COULEURS['texte_principal']),
    legend=dict(
        x=0.6,
        y=0.15,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=COULEURS['texte_secondaire'],
        borderwidth=1
    ),
    hovermode='closest'
)



modeling_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("Méthodologie de Préparation des Données", className="alert-heading mb-3"),
                html.Hr(),
                html.Div([
                    html.Strong("Division des données : "),
                    f"{nombre_exemples_train} exemples pour l'entraînement (70%), ",
                    f"{nombre_exemples_test} exemples pour le test (30%)"
                ], className="mb-2"),
                html.Div([
                    html.Strong("Stratification : "),
                    f"Taux de défaut maintenu à {taux_defaut_train:.1f}% (train) et {taux_defaut_test:.1f}% (test)"
                ], className="mb-2"),
                html.Div([
                    html.Strong("Standardisation : "),
                    "Z-score normalization (moyenne=0, écart-type=1) appliquée sur toutes les variables"
                ], className="mb-2"),
                html.Div([
                    html.Strong("Variables prédictives : "),
                    f"{len(variables_predictives)} features sélectionnées selon leur pertinence métier"
                ])
            ], color="info", className="border-0 shadow-sm")
        ], width=12, className="mb-4")
    ]),
    
    dbc.Tabs([
        dbc.Tab(
            label="Entraînement et Comparaison des Modèles",
            tab_id="modeles",
            children=[
                # Contenu du premier sous-onglet
                html.Div([
                    

        
    
    # Métriques de performance comparatives
    dbc.Row([
        # Carte LDA
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    "LDA - Linear Discriminant Analysis",
                    className="text-white fw-bold text-center",
                    style={'backgroundColor': COULEURS['navy_fonce'], 'fontSize': '1.1rem'}
                ),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.H6("Accuracy", className="text-muted mb-1 text-uppercase small"),
                            html.H3(f"{accuracy_lda:.4f}", className="fw-bold mb-0", style={'color': COULEURS['navy_fonce']})
                        ], className="text-center mb-3"),
                        
                        html.Div([
                            html.H6("F1-Score", className="text-muted mb-1 text-uppercase small"),
                            html.H3(f"{f1_score_lda:.4f}", className="fw-bold mb-0", style={'color': COULEURS['navy_fonce']})
                        ], className="text-center mb-3"),
                        
                        html.Div([
                            html.H6("Recall (Sensibilité)", className="text-muted mb-1 text-uppercase small"),
                            html.H3(f"{recall_lda:.4f}", className="fw-bold mb-0", style={'color': COULEURS['navy_fonce']})
                        ], className="text-center")
                    ])
                ], className="py-4")
            ], className="shadow-sm border-0 h-100")
        ], xs=12, md=6, className="mb-4"),
        
        # Carte QDA
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    "QDA - Quadratic Discriminant Analysis",
                    className="text-white fw-bold text-center",
                    style={'backgroundColor': COULEURS['marron_fonce'], 'fontSize': '1.1rem'}
                ),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.H6("Accuracy", className="text-muted mb-1 text-uppercase small"),
                            html.H3(f"{accuracy_qda:.4f}", className="fw-bold mb-0", style={'color': COULEURS['marron_fonce']})
                        ], className="text-center mb-3"),
                        
                        html.Div([
                            html.H6("F1-Score", className="text-muted mb-1 text-uppercase small"),
                            html.H3(f"{f1_score_qda:.4f}", className="fw-bold mb-0", style={'color': COULEURS['marron_fonce']})
                        ], className="text-center mb-3"),
                        
                        html.Div([
                            html.H6("Recall (Sensibilité)", className="text-muted mb-1 text-uppercase small"),
                            html.H3(f"{recall_qda:.4f}", className="fw-bold mb-0", style={'color': COULEURS['marron_fonce']})
                        ], className="text-center")
                    ])
                ], className="py-4")
            ], className="shadow-sm border-0 h-100")
        ], xs=12, md=6, className="mb-4")
    ]),
    
    
    # Matrices de confusion et courbe ROC
    dbc.Row([
        # Matrice de confusion LDA
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Matrice de Confusion - LDA", className="bg-white fw-bold", 
                              style={'color': COULEURS['texte_principal']}),
                dbc.CardBody([
                    dcc.Graph(
                        figure=graphique_confusion_lda,
                        config={'displayModeBar': False}
                    )
                ], className="p-2")
            ], className="shadow-sm border-0")
        ], xs=12, md=4, className="mb-4"),
        
        # Matrice de confusion QDA
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Matrice de Confusion - QDA", className="bg-white fw-bold",
                              style={'color': COULEURS['texte_principal']}),
                dbc.CardBody([
                    dcc.Graph(
                        figure=graphique_confusion_qda,
                        config={'displayModeBar': False}
                    )
                ], className="p-2")
            ], className="shadow-sm border-0")
        ], xs=12, md=4, className="mb-4"),
        
        # Courbes ROC
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Courbes ROC", className="bg-white fw-bold",
                              style={'color': COULEURS['texte_principal']}),
                dbc.CardBody([
                    dcc.Graph(
                        figure=graphique_courbes_roc,
                        config={'displayModeBar': False}
                    )
                ], className="p-2")
            ], className="shadow-sm border-0")
        ], xs=12, md=4, className="mb-4")
    ]),
    
    
    # Recommandation du meilleur modèle
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H4([
                    "Modèle Recommandé : ",
                    html.Span(nom_meilleur_modele, style={'color': couleur_meilleur})
                ], className="alert-heading"),
                html.Hr(),
                html.P([
                    f"Basé sur le F1-Score (métrique équilibrée), le modèle {nom_meilleur_modele} ",
                    f"obtient les meilleures performances avec un score de {f1_meilleur_modele:.4f}. ",
                    "Ce modèle sera utilisé pour les prédictions dans les prédictions"
                ])
            ], color="success" if nom_meilleur_modele == "LDA" else "danger", 
               className="border-0 shadow-sm text-center")
        ], width=12, className="mb-5")
    ]),
    
    
    html.Hr(className="my-5", style={'borderTop': f'3px solid {COULEURS["gris_clair"]}'}),
                ])

            ]
        ),

    dbc.Tab(
            label="Prédiction du Risque de Défaut",
            tab_id="risque_def",
            children=[
                # Contenu du deuxième sous-onglet
                html.Div([
                    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Formulaire de Saisie des Caractéristiques Client", 
                               style={'color': COULEURS['texte_principal']}),
                    html.P(
                        "Remplissez tous les champs ci-dessous pour obtenir une prédiction de risque de défaut",
                        className="text-muted small mb-0 mt-2"
                    )
                ], className="bg-light", style={'fontSize': '1.1rem'}),
                
                dbc.CardBody([
                    # Groupe 1 : Informations personnelles
                    html.H6("Informations Personnelles", 
                           className="text-uppercase mb-3 mt-3",
                           style={'color': COULEURS['texte_secondaire']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Âge du client", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-age',
                                type='number',
                                value=35,
                                min=18,
                                max=70,
                                className="mb-3"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Nombre de dépendants", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-dependants',
                                type='number',
                                value=2,
                                min=0,
                                max=10,
                                className="mb-3"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Score Mobile Money (0-100)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-mobile-money',
                                type='number',
                                value=50,
                                min=0,
                                max=100,
                                className="mb-3"
                            )
                        ], md=4)
                    ]),
                    
                    # Groupe 2 : Situation financière
                    html.H6("Situation Financière", 
                           className="text-uppercase mb-3 mt-4",
                           style={'color': COULEURS['texte_secondaire']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Revenu mensuel (FCFA)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-revenu',
                                type='number',
                                value=200000,
                                min=0,
                                step=10000,
                                className="mb-3"
                            )
                        ], md=6),
                        
                        dbc.Col([
                            html.Label("Épargne disponible (FCFA)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-epargne',
                                type='number',
                                value=100000,
                                min=0,
                                step=10000,
                                className="mb-3"
                            )
                        ], md=6)
                    ]),
                    
                    # Groupe 3 : Historique bancaire
                    html.H6("Historique Bancaire", 
                           className="text-uppercase mb-3 mt-4",
                           style={'color': COULEURS['texte_secondaire']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Ancienneté relation (mois)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-anciennete',
                                type='number',
                                value=36,
                                min=0,
                                className="mb-3"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Historique crédit (mois)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-historique',
                                type='number',
                                value=48,
                                min=0,
                                className="mb-3"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Jours de retard (12 derniers mois)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-retard',
                                type='number',
                                value=0,
                                min=0,
                                className="mb-3"
                            )
                        ], md=4)
                    ]),
                    
                    # Groupe 4 : Caractéristiques du prêt
                    html.H6("Caractéristiques du Prêt Demandé", 
                           className="text-uppercase mb-3 mt-4",
                           style={'color': COULEURS['texte_secondaire']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Montant du prêt (FCFA)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-montant-pret',
                                type='number',
                                value=300000,
                                min=0,
                                step=10000,
                                className="mb-3"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Durée du prêt (mois)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-duree',
                                type='number',
                                value=12,
                                min=1,
                                max=60,
                                className="mb-3"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Taux d'intérêt annuel (%)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-taux',
                                type='number',
                                value=28.0,
                                min=0,
                                max=50,
                                step=0.1,
                                className="mb-3"
                            )
                        ], md=4)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Ratio DSTI (%)", className="fw-bold mb-2"),
                            dbc.Input(
                                id='input-dsti',
                                type='number',
                                value=15.0,
                                min=0,
                                max=100,
                                step=0.1,
                                className="mb-3"
                            )
                        ], md=6),
                        
                        dbc.Col([
                            html.Label("Type de prêt", className="fw-bold mb-2"),
                            dbc.RadioItems(
                                id='input-pret-groupe',
                                options=[
                                    {'label': 'Prêt individuel', 'value': 0},
                                    {'label': 'Prêt de groupe', 'value': 1}
                                ],
                                value=0,
                                className="mb-3"
                            )
                        ], md=6)
                    ]),
                    
                    # Bouton de prédiction
                    html.Div([
                        dbc.Button(
                            "Calculer les Probabilités de Défaut",
                            id='predict-button',
                            color="primary",
                            size="lg",
                            className="w-100 mt-4",
                            style={'fontSize': '1.2rem', 'fontWeight': 'bold', 
                                  'backgroundColor': COULEURS['navy_fonce'], 'borderColor': COULEURS['navy_fonce']}
                        )
                    ])
                ], className="p-4")
            ], className="shadow border-0")
        ], width=12, className="mb-4")
    ]),
    
    
    # Zone d'affichage des résultats de prédiction
    dbc.Row([
        dbc.Col([
            html.Div(id='prediction-results')
        ], width=12)
    ], className="mb-5"),
                ])
            ]
        ),
 ])

    ], fluid=True)