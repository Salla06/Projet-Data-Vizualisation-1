import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
# CHARGEMENT DES DONNÉES
# ----------------------------------------------------------------------------
# Importation du fichier de données client de microfinance
df = pd.read_excel('Data/microfinance_credit_risk.xlsx')




analysis_layout = dbc.Container([
    # ------------------------------------------------------------------------
    # SECTION 1 : EN-TÊTE DE LA PAGE
    # ------------------------------------------------------------------------
    # SECTION 2 : PANNEAU DE FILTRES INTERACTIFS
    # ------------------------------------------------------------------------
    # Système de filtrage multi-critères permettant de segmenter les données
    # selon 6 dimensions : région, secteur, canal, statut, montant, DSTI
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-filter me-2"),
                    html.Strong("Filtres de Segmentation"),
                ], className="bg-light", style={'fontSize': '1.1rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    # Ligne 1 : Filtres géographiques et sectoriels
                    dbc.Row([
                        # Filtre : Région géographique
                        dbc.Col([
                            html.Label(["Région"
                            ], className="fw-bold mb-2", style={'color': COULEURS['texte_principal']}),
                            dcc.Dropdown(
                                id='filter-region',
                                options=[{'label': 'Toutes les régions', 'value': 'ALL'}] + 
                                        [{'label': region, 'value': region} 
                                         for region in sorted(df['region'].unique())],
                                value='ALL',
                                clearable=False,
                                className="shadow-sm"
                            )
                        ], md=3, className="mb-3"),
                        
                        # Filtre : Secteur d'activité
                        dbc.Col([
                            html.Label([
                                
                                "Secteur d'activité"
                            ], className="fw-bold mb-2", style={'color': COULEURS['texte_principal']}),
                            dcc.Dropdown(
                                id='filter-sector',
                                options=[{'label': 'Tous les secteurs', 'value': 'ALL'}] + 
                                        [{'label': secteur, 'value': secteur} 
                                         for secteur in sorted(df['secteur_activite'].unique())],
                                value='ALL',
                                clearable=False,
                                className="shadow-sm"
                            )
                        ], md=3, className="mb-3"),
                        
                        # Filtre : Canal de distribution
                        dbc.Col([
                            html.Label([
                               "Canal de distribution"
                            ], className="fw-bold mb-2", style={'color': COULEURS['texte_principal']}),
                            dcc.Dropdown(
                                id='filter-channel',
                                options=[{'label': 'Tous les canaux', 'value': 'ALL'}] + 
                                        [{'label': canal, 'value': canal} 
                                         for canal in sorted(df['canal_octroi'].unique())],
                                value='ALL',
                                clearable=False,
                                className="shadow-sm"
                            )
                        ], md=3, className="mb-3"),
                        
                        # Filtre : Statut de défaut
                        dbc.Col([
                            html.Label([
                                "Statut de défaut"
                            ], className="fw-bold mb-2", style={'color': COULEURS['texte_principal']}),
                            dcc.Dropdown(
                                id='filter-default',
                                options=[
                                    {'label': 'Tous les clients', 'value': 'ALL'},
                                    {'label': 'Sans défaut', 'value': 0},
                                    {'label': 'En défaut', 'value': 1}
                                ],
                                value='ALL',
                                clearable=False,
                                className="shadow-sm"
                            )
                        ], md=3, className="mb-3")
                    ]),
                    
                    # Ligne 2 : Filtres numériques (sliders)
                    dbc.Row([
                        # Filtre : Montant du prêt
                        dbc.Col([
                            html.Label([
                                "Montant du prêt (FCFA)"
                            ], className="fw-bold mb-3", style={'color': COULEURS['texte_principal']}),
                            dcc.RangeSlider(
                                id='filter-loan-amount',
                                min=df['montant_pret_xof'].min(),
                                max=df['montant_pret_xof'].max(),
                                value=[
                                    df['montant_pret_xof'].min(),
                                    df['montant_pret_xof'].max()
                                ],
                                marks={
                                    int(df['montant_pret_xof'].min()): {
                                        'label': f"{int(df['montant_pret_xof'].min()/1000)}K",
                                        'style': {'color': COULEURS['texte_secondaire']}
                                    },
                                    int(df['montant_pret_xof'].quantile(0.5)): {
                                        'label': f"{int(df['montant_pret_xof'].quantile(0.5)/1000)}K",
                                        'style': {'color': COULEURS['texte_secondaire']}
                                    },
                                    int(df['montant_pret_xof'].max()): {
                                        'label': f"{int(df['montant_pret_xof'].max()/1000)}K",
                                        'style': {'color': COULEURS['texte_secondaire']}
                                    }
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3"
                            )
                        ], md=6),
                        
                        # Filtre : Ratio DSTI
                        dbc.Col([
                            html.Label([
                                "Ratio DSTI (%)"
                            ], className="fw-bold mb-3", style={'color': COULEURS['texte_principal']}),
                            dcc.RangeSlider(
                                id='filter-dsti',
                                min=df['dsti_pct'].min(),
                                max=df['dsti_pct'].max(),
                                value=[
                                    df['dsti_pct'].min(),
                                    df['dsti_pct'].max()
                                ],
                                marks={
                                    int(df['dsti_pct'].min()): {
                                        'label': f"{int(df['dsti_pct'].min())}%",
                                        'style': {'color': COULEURS['texte_secondaire']}
                                    },
                                    int(df['dsti_pct'].quantile(0.5)): {
                                        'label': f"{int(df['dsti_pct'].quantile(0.5))}%",
                                        'style': {'color': COULEURS['texte_secondaire']}
                                    },
                                    int(df['dsti_pct'].max()): {
                                        'label': f"{int(df['dsti_pct'].max())}%",
                                        'style': {'color': COULEURS['texte_secondaire']}
                                    }
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3"
                            )
                        ], md=6)
                    ])
                ], className="p-4")
            ], className="shadow-sm border-0 mb-4")
        ])
    ]),
    # Sous-onglets
    dbc.Tabs([
        # Premier sous-onglet
        dbc.Tab(
            label="Vue d'ensemble",
            tab_id="vue-ensemble",
            children=[
                # Contenu du premier sous-onglet
                html.Div([
                        # ------------------------------------------------------------------------
    
    dbc.Row([
        # KPI 1 : Nombre de clients filtrés
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-users", style={'fontSize': '1.3rem', 'color': COULEURS['vert_fonce']}),
                        html.H3(id='kpi-total-clients', className="fw-bold mt-3 mb-1", 
                               style={'color': COULEURS['texte_principal']}),
                        html.P("Clients", className="text-muted mb-0 small text-uppercase")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'borderLeft': f'4px solid {COULEURS["vert_fonce"]}'})
        ], xs=6, md=2, className="mb-3"),
        
        # KPI 2 : Taux de défaut du segment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-percentage", style={'fontSize': '1.3rem', 'color': COULEURS['marron_fonce']}),
                        html.H3(id='kpi-default-rate', className="fw-bold mt-3 mb-1", 
                               style={'color': COULEURS['texte_principal']}),
                        html.P("Taux défaut", className="text-muted mb-0 small text-uppercase")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'borderLeft': f'4px solid {COULEURS["marron_fonce"]}'})
        ], xs=6, md=2, className="mb-3"),
        
        # KPI 3 : Nombre de régions dans le segment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-map-marked-alt", style={'fontSize': '1.3rem', 'color': COULEURS['marron_secondaire']}),
                        html.H3(id='kpi-regions', className="fw-bold mt-3 mb-1", 
                               style={'color': COULEURS['texte_principal']}),
                        html.P("Régions", className="text-muted mb-0 small text-uppercase")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'borderLeft': f'4px solid {COULEURS["marron_secondaire"]}'})
        ], xs=6, md=2, className="mb-3"),
        
        # KPI 4 : Nombre de secteurs dans le segment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-briefcase", style={'fontSize': '1.3rem', 'color': COULEURS['dore']}),
                        html.H3(id='kpi-sectors', className="fw-bold mt-3 mb-1", 
                               style={'color': COULEURS['texte_principal']}),
                        html.P("Secteurs", className="text-muted mb-0 small text-uppercase")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'borderLeft': f'4px solid {COULEURS["dore"]}'})
        ], xs=6, md=2, className="mb-3"),
        
        # KPI 5 : Nombre de canaux dans le segment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-route", style={'fontSize': '1.3rem', 'color': COULEURS['navy_fonce']}),
                        html.H3(id='kpi-channels', className="fw-bold mt-3 mb-1", 
                               style={'color': COULEURS['texte_principal']}),
                        html.P("Canaux", className="text-muted mb-0 small text-uppercase")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'borderLeft': f'4px solid {COULEURS["navy_fonce"]}'})
        ], xs=6, md=2, className="mb-3"),
        
        # KPI 6 : DSTI moyen du segment
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-calculator", style={'fontSize': '1.3rem', 'color': COULEURS['texte_principal']}),
                        html.H3(id='kpi-avg-dsti', className="fw-bold mt-3 mb-1", 
                               style={'color': COULEURS['texte_principal']}),
                        html.P("DSTI moyen", className="text-muted mb-0 small text-uppercase")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'borderLeft': f'4px solid {COULEURS["texte_principal"]}'})
        ], xs=6, md=2, className="mb-3")
    ], className="mb-4"),
    
    
    # ------------------------------------------------------------------------
    # SECTION 4 : ANALYSE APPROFONDIE DU DSTI
    # ------------------------------------------------------------------------
    # Le DSTI (Debt Service To Income) est un indicateur critique de risque
    # Trois modes de visualisation pour analyser sa distribution
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-line me-2"),
                    html.Strong("Analyse du DSTI (Debt Service to Income)"),
                    
                ], className="bg-white", style={'fontSize': '1.1rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    # Onglets pour différents types de visualisation
                    dbc.Tabs([
                        dbc.Tab(
                            label="Histogramme",
                            tab_id="hist",
                            label_style={'cursor': 'pointer'}
                        ),
                        dbc.Tab(
                            label="Box Plot",
                            tab_id="box",
                            label_style={'cursor': 'pointer'}
                        ),
                        dbc.Tab(
                            label="Violin Plot",
                            tab_id="violin",
                            label_style={'cursor': 'pointer'}
                        )
                    ], id="dsti-tabs", active_tab="hist", className="mb-3"),
                    
                    # Conteneur pour le graphique qui changera selon l'onglet sélectionné
                    html.Div(id="dsti-plot-container")
                ], className="p-3")
            ], className="shadow-sm border-0")
        ], width=12, className="mb-4")
    ]),
    
    
    # ------------------------------------------------------------------------
    # SECTION 5 : DIAGRAMME SANKEY - FLUX DU PORTEFEUILLE
    # ------------------------------------------------------------------------
    # Innovation : Visualisation du parcours client et des flux de portefeuille
    # Cette visualisation révèle les patterns de comportement et de risque
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-project-diagram me-2"),
                    html.Strong("Flux du Portefeuille"),
                ], className="bg-white", style={'fontSize': '1.1rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    dcc.Graph(
                        id='sankey-portfolio-flow',
                        config={'displayModeBar': False},
                        style={'height': '450px'}
                    )
                ], className="p-3")
            ], className="shadow-sm border-0")
        ], width=12, className="mb-4")
    ]),
    
                ])
            ]
        ),
        
        # Deuxième sous-onglet
        dbc.Tab(
            label="Analyses & Tendances",
            tab_id="analyses-tendances",
            children=[
                # Contenu du deuxième sous-onglet
                html.Div([
                    # ------------------------------------------------------------------------
    # SECTION 6 : ANALYSE MULTIVARIÉE INTERACTIVE
    # ------------------------------------------------------------------------
    # Scatter plot entièrement personnalisable pour explorer les relations
    # entre n'importe quelles variables du dataset
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-scatter me-2"),
                    html.Strong("Exploration Multivariée Interactive"),
                    
                ], className="bg-white", style={'fontSize': '1.1rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    # Contrôles pour personnaliser le scatter plot
                    dbc.Row([
                        # Sélection axe X
                        dbc.Col([
                            html.Label("Axe horizontal (X)", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='scatter-x-axis',
                                options=[
                                    {'label': 'Revenu mensuel', 'value': 'revenu_mensuel_xof'},
                                    {'label': 'Montant du prêt', 'value': 'montant_pret_xof'},
                                    {'label': 'DSTI (%)', 'value': 'dsti_pct'},
                                    {'label': 'Âge', 'value': 'age'},
                                    {'label': 'Durée (mois)', 'value': 'duree_mois'},
                                    {'label': 'Taux d\'intérêt', 'value': 'taux_interet_annuel_pct'},
                                    {'label': 'Épargne', 'value': 'epargne_xof'},
                                    {'label': 'Nb dépendants', 'value': 'nb_dependants'},
                                    {'label': 'Jours de retard (12m)', 'value': 'jours_retard_12m'}
                                ],
                                value='revenu_mensuel_xof',
                                clearable=False
                            )
                        ], md=4, className="mb-3"),
                        
                        # Sélection axe Y
                        dbc.Col([
                            html.Label("Axe vertical (Y)", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='scatter-y-axis',
                                options=[
                                    {'label': 'Revenu mensuel', 'value': 'revenu_mensuel_xof'},
                                    {'label': 'Montant du prêt', 'value': 'montant_pret_xof'},
                                    {'label': 'DSTI (%)', 'value': 'dsti_pct'},
                                    {'label': 'Âge', 'value': 'age'},
                                    {'label': 'Durée (mois)', 'value': 'duree_mois'},
                                    {'label': 'Taux d\'intérêt', 'value': 'taux_interet_annuel_pct'},
                                    {'label': 'Épargne', 'value': 'epargne_xof'},
                                    {'label': 'Nb dépendants', 'value': 'nb_dependants'},
                                    {'label': 'Jours de retard (12m)', 'value': 'jours_retard_12m'}
                                ],
                                value='montant_pret_xof',
                                clearable=False
                            )
                        ], md=4, className="mb-3"),
                        
                        # Sélection dimension de couleur
                        dbc.Col([
                            html.Label("Dimension de couleur", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='scatter-color',
                                options=[
                                    {'label': 'Statut de défaut', 'value': 'defaut_90j'},
                                    {'label': 'Région', 'value': 'region'},
                                    {'label': 'Secteur', 'value': 'secteur_activite'},
                                    {'label': 'Canal', 'value': 'canal_octroi'}
                                ],
                                value='defaut_90j',
                                clearable=False
                            )
                        ], md=4, className="mb-3")
                    ]),
                    
                    # Graphique scatter plot
                    dcc.Graph(
                        id='scatter-plot',
                        config={'displayModeBar': True},
                        style={'height': '500px'}
                    )
                ], className="p-3")
            ], className="shadow-sm border-0")
        ], width=12, className="mb-4")
    ]),
    
    
    # ------------------------------------------------------------------------
    # SECTION 7 : VISUALISATIONS COMPARATIVES
    # ------------------------------------------------------------------------
    # Deux analyses côte à côte : Corrélations et Analyse par catégorie
    
    dbc.Row([
        # Matrice de corrélation
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-grip-horizontal me-2"),
                    "Matrice de Corrélation"
                ], className="bg-white fw-bold", style={'fontSize': '1.05rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    dcc.Graph(
                        id='correlation-matrix',
                        config={'displayModeBar': False},
                        style={'height': '450px'}
                    )
                ], className="p-3")
            ], className="shadow-sm border-0 h-100")
        ], xs=12, lg=6, className="mb-4"),
        
        # Analyse par catégorie
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Taux de Défaut par Catégorie"
                ], className="bg-white fw-bold", style={'fontSize': '1.05rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    dcc.Graph(
                        id='default-by-category',
                        config={'displayModeBar': False},
                        style={'height': '450px'}
                    )
                ], className="p-3")
            ], className="shadow-sm border-0 h-100")
        ], xs=12, lg=6, className="mb-4")
    ]),
    
    
    # ------------------------------------------------------------------------
    # SECTION 8 : ANALYSE DE DISTRIBUTION COMPARATIVE
    # ------------------------------------------------------------------------
    # Distribution des montants de prêts avec comparaison défaut/non-défaut
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-money-check-alt me-2"),
                    html.Strong("Distribution des Montants de Prêts"),
                    
                ], className="bg-white", style={'fontSize': '1.1rem', 'color': COULEURS['texte_principal']}),
                
                dbc.CardBody([
                    dcc.Graph(
                        id='loan-distribution',
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ], className="p-3")
            ], className="shadow-sm border-0")
        ], width=12, className="mb-5")
    ]),
    
                ])
            ]
        ),
        
        # Troisième sous-onglet
        dbc.Tab(
            label="Détails Projets",
            tab_id="details-projets",
            children=[
                # Contenu du troisième sous-onglet
                html.Div([
                    # ------------------------------------------------------------------------
    # SECTION 9 : TABLE DE DONNÉES INTERACTIVE
    # ------------------------------------------------------------------------
    # Table complète avec capacités de tri, filtrage et recherche
    # Les données affichées correspondent aux filtres actifs
    
    html.H4(
        className="mb-3",
        style={'color': COULEURS['texte_principal'], 'fontWeight': '600'}
    ),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-table me-2"),
                    html.Strong("Table Interactive des Clients"),
                    
                    
                ], className="bg-light", style={'fontSize': '1.1rem', 'color': COULEURS['navy_fonce']}),
                
                dbc.CardBody([
                    dash_table.DataTable(
                        id='data-table',
                        columns=[
                            {"name": col, "id": col} 
                            for col in df.columns
                        ],
                        page_size=15,
                        page_action='native',
                        
                        # Activation du tri sur toutes les colonnes
                        sort_action="native",
                        sort_mode="multi",
                        
                        # Activation du filtrage
                        filter_action="native",
                        
                        # Style de la table
                        style_table={
                            'overflowX': 'auto',
                            'minWidth': '100%'
                        },
                        
                        # Style des cellules
                        style_cell={
                            'textAlign': 'left',
                            'padding': '12px',
                            'fontSize': '13px',
                            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                        },
                        
                        # Style de l'en-tête
                        style_header={
                            'backgroundColor': COULEURS['navy_fonce'],
                            'color': COULEURS['blanc'],
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'fontSize': '14px'
                        },
                        
                        # Mise en forme conditionnelle
                        style_data_conditional=[
                            # Lignes alternées pour meilleure lisibilité
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': COULEURS['gris_clair']
                            },
                            # Mise en évidence des clients en défaut
                            {
                                'if': {
                                    'filter_query': '{defaut_90j} = 1',
                                    'column_id': 'defaut_90j'
                                },
                                'backgroundColor': '#ffe6e6',
                                'color': '#c62828',
                                'fontWeight': 'bold'
                            },
                            # Mise en évidence des DSTI élevés (> 50%)
                            {
                                'if': {
                                    'filter_query': '{dsti_pct} > 50',
                                    'column_id': 'dsti_pct'
                                },
                                'backgroundColor': '#fff3cd',
                                'color': '#856404'
                            }
                        ],
                        
                        # Style au survol
                        style_cell_conditional=[
                            {
                                'if': {'column_id': 'client_id'},
                                'fontWeight': 'bold',
                                'color': COULEURS['texte_principal']
                            }
                        ]
                    )
                ], className="p-3")
            ], className="shadow-sm border-0")
        ])
    ], className="mb-5"),
    
                ])
            ]
        )
    ], id="analysis-tabs", active_tab="vue-ensemble")
    
], fluid=True)