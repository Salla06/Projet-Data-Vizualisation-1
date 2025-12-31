from dash import Input, Output, State, callback, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from page_modeling import (
    modele_lda, modele_qda, standardiseur,
    nom_meilleur_modele, f1_meilleur_modele,
    variables_predictives
)

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

# Load data
df = pd.read_excel('Data/microfinance_credit_risk.xlsx')

# Page routing callback
from app import app
from page_home import home_layout
from page_analysis import analysis_layout
from page_modeling import modeling_layout

@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Route between different pages"""
    if pathname == '/analysis':
        return analysis_layout
    elif pathname == '/modeling':
        return modeling_layout
    else:
        return home_layout

# ============================================================================
# ANALYSIS PAGE CALLBACKS
# ============================================================================

@callback(
    [Output('kpi-total-clients', 'children'),
     Output('kpi-default-rate', 'children'),
     Output('kpi-regions', 'children'),
     Output('kpi-sectors', 'children'),
     Output('kpi-channels', 'children'),
     Output('kpi-avg-dsti', 'children'),
     Output('data-table', 'data'),
     Output('correlation-matrix', 'figure'),
     Output('default-by-category', 'figure'),
     Output('loan-distribution', 'figure')],
    [Input('filter-region', 'value'),
     Input('filter-sector', 'value'),
     Input('filter-channel', 'value'),
     Input('filter-default', 'value'),
     Input('filter-loan-amount', 'value'),
     Input('filter-dsti', 'value')]
)
def update_analysis_page(region, sector, channel, default, loan_amount, dsti):
    """Update all components on the analysis page based on filters"""
    # Filter data
    filtered_df = df.copy()
    
    if region != 'ALL':
        filtered_df = filtered_df[filtered_df['region'] == region]
    if sector != 'ALL':
        filtered_df = filtered_df[filtered_df['secteur_activite'] == sector]
    if channel != 'ALL':
        filtered_df = filtered_df[filtered_df['canal_octroi'] == channel]
    if default != 'ALL':
        filtered_df = filtered_df[filtered_df['defaut_90j'] == default]
    
    filtered_df = filtered_df[
        (filtered_df['montant_pret_xof'] >= loan_amount[0]) &
        (filtered_df['montant_pret_xof'] <= loan_amount[1]) &
        (filtered_df['dsti_pct'] >= dsti[0]) &
        (filtered_df['dsti_pct'] <= dsti[1])
    ]
    
    # Calculate KPIs
    total_clients = len(filtered_df)
    default_rate = (filtered_df['defaut_90j'].sum() / total_clients * 100) if total_clients > 0 else 0
    n_regions = filtered_df['region'].nunique()
    n_sectors = filtered_df['secteur_activite'].nunique()
    n_channels = filtered_df['canal_octroi'].nunique()
    avg_dsti = filtered_df['dsti_pct'].mean()
    
    # Correlation matrix
    numeric_cols = ['age', 'revenu_mensuel_xof', 'epargne_xof', 'montant_pret_xof', 
                    'duree_mois', 'taux_interet_annuel_pct', 'dsti_pct', 
                    'jours_retard_12m', 'nb_dependants', 'defaut_90j']
    corr = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale=[[0, COULEURS['navy_fonce']], [0.5, COULEURS['blanc']], [1, COULEURS['marron_fonce']]],
        zmin=-1, zmax=1,
        labels=dict(color="Corrélation")
    )
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        font=dict(color=COULEURS['texte_principal'])
    )
    
    # Default by category
    default_by_region = filtered_df.groupby('region')['defaut_90j'].agg(['sum', 'count'])
    default_by_region['rate'] = (default_by_region['sum'] / default_by_region['count'] * 100)
    
    fig_default = px.bar(
        x=default_by_region.index,
        y=default_by_region['rate'],
        title='Taux de défaut par région',
        labels={'x': 'Région', 'y': 'Taux de défaut (%)'},
        color=default_by_region['rate'],
        color_continuous_scale=[[0, COULEURS['vert_fonce']], [0.5, COULEURS['dore']], [1, COULEURS['marron_fonce']]]
    )
    fig_default.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(color=COULEURS['texte_principal'])
    )
    
    # Loan distribution
    fig_loan = px.histogram(
        filtered_df,
        x='montant_pret_xof',
        nbins=30,
        title='Distribution des montants de prêts',
        labels={'montant_pret_xof': 'Montant du prêt (XOF)', 'count': 'Fréquence'},
        color_discrete_sequence=[COULEURS['navy_fonce']]
    )
    fig_loan.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(color=COULEURS['texte_principal'])
    )
    
    return (
        f"{total_clients:,}",
        f"{default_rate:.2f}%",
        str(n_regions),
        str(n_sectors),
        str(n_channels),
        f"{avg_dsti:.1f}%",
        filtered_df.to_dict('records'),
        fig_corr,
        fig_default,
        fig_loan
    )

@callback(
    Output('dsti-plot-container', 'children'),
    [Input('dsti-tabs', 'active_tab'),
     Input('filter-region', 'value'),
     Input('filter-sector', 'value'),
     Input('filter-channel', 'value'),
     Input('filter-default', 'value'),
     Input('filter-loan-amount', 'value'),
     Input('filter-dsti', 'value')]
)
def update_dsti_plot(active_tab, region, sector, channel, default, loan_amount, dsti):
    """Update DSTI plot based on selected tab and filters"""
    # Filter data
    filtered_df = df.copy()
    
    if region != 'ALL':
        filtered_df = filtered_df[filtered_df['region'] == region]
    if sector != 'ALL':
        filtered_df = filtered_df[filtered_df['secteur_activite'] == sector]
    if channel != 'ALL':
        filtered_df = filtered_df[filtered_df['canal_octroi'] == channel]
    if default != 'ALL':
        filtered_df = filtered_df[filtered_df['defaut_90j'] == default]
    
    filtered_df = filtered_df[
        (filtered_df['montant_pret_xof'] >= loan_amount[0]) &
        (filtered_df['montant_pret_xof'] <= loan_amount[1]) &
        (filtered_df['dsti_pct'] >= dsti[0]) &
        (filtered_df['dsti_pct'] <= dsti[1])
    ]
    
    # Create appropriate plot based on tab
    if active_tab == 'hist':
        fig = px.histogram(
            filtered_df,
            x='dsti_pct',
            nbins=30,
            color='defaut_90j',
            barmode='overlay',
            labels={'dsti_pct': 'DSTI (%)', 'defaut_90j': 'Défaut'},
            color_discrete_map={0: COULEURS['navy_fonce'], 1: COULEURS['marron_fonce']}
        )
        fig.update_traces(opacity=0.7)
    elif active_tab == 'box':
        filtered_df['Statut'] = filtered_df['defaut_90j'].map({0: 'Non Défaut', 1: 'Défaut'})
        fig = px.box(
            filtered_df,
            x='Statut',
            y='dsti_pct',
            color='Statut',
            labels={'dsti_pct': 'DSTI (%)'},
            color_discrete_map={'Non Défaut': COULEURS['navy_fonce'], 'Défaut': COULEURS['marron_fonce']}
        )
    else:  # violin
        filtered_df['Statut'] = filtered_df['defaut_90j'].map({0: 'Non Défaut', 1: 'Défaut'})
        fig = px.violin(
            filtered_df,
            x='Statut',
            y='dsti_pct',
            color='Statut',
            labels={'dsti_pct': 'DSTI (%)'},
            color_discrete_map={'Non Défaut': COULEURS['navy_fonce'], 'Défaut': COULEURS['marron_fonce']},
            box=True
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(color=COULEURS['texte_principal'])
    )
    
    from dash import dcc
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

@callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-axis', 'value'),
     Input('scatter-y-axis', 'value'),
     Input('scatter-color', 'value'),
     Input('filter-region', 'value'),
     Input('filter-sector', 'value'),
     Input('filter-channel', 'value'),
     Input('filter-default', 'value'),
     Input('filter-loan-amount', 'value'),
     Input('filter-dsti', 'value')]
)
def update_scatter_plot(x_axis, y_axis, color_by, region, sector, channel, default, loan_amount, dsti):
    """Update scatter plot based on user selection"""
    # Filter data
    filtered_df = df.copy()
    
    if region != 'ALL':
        filtered_df = filtered_df[filtered_df['region'] == region]
    if sector != 'ALL':
        filtered_df = filtered_df[filtered_df['secteur_activite'] == sector]
    if channel != 'ALL':
        filtered_df = filtered_df[filtered_df['canal_octroi'] == channel]
    if default != 'ALL':
        filtered_df = filtered_df[filtered_df['defaut_90j'] == default]
    
    filtered_df = filtered_df[
        (filtered_df['montant_pret_xof'] >= loan_amount[0]) &
        (filtered_df['montant_pret_xof'] <= loan_amount[1]) &
        (filtered_df['dsti_pct'] >= dsti[0]) &
        (filtered_df['dsti_pct'] <= dsti[1])
    ]
    
    # Create scatter plot
    if color_by == 'defaut_90j':
        filtered_df['Statut'] = filtered_df['defaut_90j'].map({0: 'Non Défaut', 1: 'Défaut'})
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color='Statut',
            color_discrete_map={'Non Défaut': COULEURS['vert_fonce'], 'Défaut': COULEURS['marron_fonce']},
            opacity=0.6,
            hover_data=['client_id']
        )
    else:
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            opacity=0.6,
            hover_data=['client_id'],
            color_discrete_sequence=[COULEURS['navy_fonce'], COULEURS['dore'], COULEURS['marron_secondaire'], COULEURS['vert_fonce']]
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        font=dict(color=COULEURS['texte_principal'])
    )
    
    return fig

@callback(
    Output('sankey-portfolio-flow', 'figure'),
    [Input('filter-region', 'value'),
     Input('filter-sector', 'value'),
     Input('filter-channel', 'value'),
     Input('filter-default', 'value'),
     Input('filter-loan-amount', 'value'),
     Input('filter-dsti', 'value')]
)
def update_sankey_diagram(region, sector, channel, default, loan_amount, dsti):
    """
    Crée un diagramme Sankey montrant le flux du portefeuille
    Région → Secteur → Canal → Statut de défaut
    """
    # Filtrer les données
    filtered_df = df.copy()
    
    if region != 'ALL':
        filtered_df = filtered_df[filtered_df['region'] == region]
    if sector != 'ALL':
        filtered_df = filtered_df[filtered_df['secteur_activite'] == sector]
    if channel != 'ALL':
        filtered_df = filtered_df[filtered_df['canal_octroi'] == channel]
    if default != 'ALL':
        filtered_df = filtered_df[filtered_df['defaut_90j'] == default]
    
    filtered_df = filtered_df[
        (filtered_df['montant_pret_xof'] >= loan_amount[0]) &
        (filtered_df['montant_pret_xof'] <= loan_amount[1]) &
        (filtered_df['dsti_pct'] >= dsti[0]) &
        (filtered_df['dsti_pct'] <= dsti[1])
    ]
    
    # Créer les flux : Région → Secteur → Statut
    flux_region_secteur = filtered_df.groupby(['region', 'secteur_activite']).size().reset_index(name='count')
    flux_secteur_statut = filtered_df.groupby(['secteur_activite', 'defaut_90j']).size().reset_index(name='count')
    
    # Créer les labels uniques pour tous les noeuds
    regions = filtered_df['region'].unique().tolist()
    secteurs = filtered_df['secteur_activite'].unique().tolist()
    statuts = ['Non Défaut', 'Défaut']
    
    all_labels = regions + secteurs + statuts
    
    # Créer un mapping label -> index
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # Préparer les sources, targets et values pour le Sankey
    sources = []
    targets = []
    values = []
    colors = []
    
    # Flux Région → Secteur
    for _, row in flux_region_secteur.iterrows():
        sources.append(label_to_idx[row['region']])
        targets.append(label_to_idx[row['secteur_activite']])
        values.append(row['count'])
        colors.append(f'rgba({int(COULEURS["navy_fonce"][1:3], 16)}, {int(COULEURS["navy_fonce"][3:5], 16)}, {int(COULEURS["navy_fonce"][5:7], 16)}, 0.4)')
    
    # Flux Secteur → Statut
    for _, row in flux_secteur_statut.iterrows():
        statut_label = 'Défaut' if row['defaut_90j'] == 1 else 'Non Défaut'
        sources.append(label_to_idx[row['secteur_activite']])
        targets.append(label_to_idx[statut_label])
        values.append(row['count'])
        # Couleur selon le statut
        if row['defaut_90j'] == 1:
            color = f'rgba({int(COULEURS["marron_fonce"][1:3], 16)}, {int(COULEURS["marron_fonce"][3:5], 16)}, {int(COULEURS["marron_fonce"][5:7], 16)}, 0.4)'
        else:
            color = f'rgba({int(COULEURS["vert_fonce"][1:3], 16)}, {int(COULEURS["vert_fonce"][3:5], 16)}, {int(COULEURS["vert_fonce"][5:7], 16)}, 0.4)'
        colors.append(color)
    
    # Couleurs des noeuds
    node_colors = (
        [COULEURS['navy_fonce']] * len(regions) +
        [COULEURS['dore']] * len(secteurs) +
        [COULEURS['vert_fonce'], COULEURS['marron_fonce']]
    )
    
    # Créer le diagramme Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color=COULEURS['blanc'], width=2),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])
    
    fig.update_layout(
        title={
            'text': 'Flux : Région → Secteur → Statut de Défaut',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': COULEURS['texte_principal']}
        },
        font=dict(size=12, color=COULEURS['texte_principal']),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450
    )
    
    return fig

# ============================================================================
# MODELING PAGE CALLBACKS
# ============================================================================
# Note : Les graphiques de la page modélisation sont créés directement
# dans page_modeling.py (pas de callback nécessaire car statiques)
# Seul le callback de prédiction est interactif

@callback(
    Output('prediction-results', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('input-age', 'value'),
     State('input-revenu', 'value'),
     State('input-epargne', 'value'),
     State('input-dependants', 'value'),
     State('input-anciennete', 'value'),
     State('input-historique', 'value'),
     State('input-retard', 'value'),
     State('input-mobile-money', 'value'),
     State('input-montant-pret', 'value'),
     State('input-duree', 'value'),
     State('input-taux', 'value'),
     State('input-dsti', 'value'),
     State('input-pret-groupe', 'value')],
    prevent_initial_call=True
)
def calculer_prediction_defaut(n_clicks, age, revenu, epargne, dependants, anciennete, 
                                historique, retard, mobile_money, montant_pret, duree, 
                                taux, dsti, pret_groupe):
    """
    Callback pour la prédiction du risque de défaut d'un nouveau client
    
    Processus :
    1. Validation des inputs
    2. Construction du vecteur de features
    3. Standardisation avec les paramètres appris sur l'ensemble train
    4. Prédiction avec LDA et QDA
    5. Affichage des résultats avec recommandation
    """
    
    # Validation : vérifier que tous les champs sont remplis
    if None in [age, revenu, epargne, dependants, anciennete, historique, 
                retard, mobile_money, montant_pret, duree, taux, dsti, pret_groupe]:
        return dbc.Alert(
            "Veuillez remplir tous les champs du formulaire avant de lancer la prédiction.",
            color="warning",
            className="text-center"
        )
    
    # Construction du vecteur de features dans le même ordre que l'entraînement
    # L'ordre doit correspondre exactement à variables_predictives
    vecteur_nouveau_client = np.array([[
        age, revenu, epargne, anciennete, historique, retard, 
        dependants, mobile_money, montant_pret, duree, taux, dsti, pret_groupe
    ]])
    
    # Standardisation avec les mêmes paramètres que l'ensemble train
    vecteur_nouveau_client_standardise = standardiseur.transform(vecteur_nouveau_client)
    
    # Calcul des probabilités de défaut avec les deux modèles
    probabilite_defaut_lda = modele_lda.predict_proba(vecteur_nouveau_client_standardise)[0][1]
    probabilite_defaut_qda = modele_qda.predict_proba(vecteur_nouveau_client_standardise)[0][1]
    
    # Décision basée sur le meilleur modèle identifié
    if nom_meilleur_modele == "LDA":
        probabilite_recommandee = probabilite_defaut_lda
        couleur_modele_recommande = COULEURS['navy_fonce']
    else:
        probabilite_recommandee = probabilite_defaut_qda
        couleur_modele_recommande = COULEURS['marron_fonce']
    
    # Classification binaire : seuil à 50%
    if probabilite_recommandee > 0.5:
        decision_finale = "DÉFAUT PROBABLE"
        couleur_decision = "danger"
        icone_decision = "exclamation-triangle"
        recommandation = "Risque élevé de défaut. Analyse approfondie recommandée avant octroi."
    else:
        decision_finale = "REMBOURSEMENT PROBABLE"
        couleur_decision = "success"
        icone_decision = "check-circle"
        recommandation = "Profil acceptable. Le client présente un risque de défaut faible."
    
    # Construction de l'affichage des résultats
    resultats = html.Div([
        
        # Titre de la section résultats
        html.H4(
            "Résultats de l'Analyse Prédictive",
            className="text-center mb-4 mt-3",
            style={'color': COULEURS['texte_principal'], 'fontWeight': '600'}
        ),
        
        # Cartes de probabilités pour chaque modèle
        dbc.Row([
            # Carte LDA
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        "LDA - Linear Discriminant Analysis",
                        className="text-white text-center fw-bold",
                        style={'backgroundColor': COULEURS['navy_fonce']}
                    ),
                    dbc.CardBody([
                        html.H2(
                            f"{probabilite_defaut_lda*100:.2f}%",
                            className="text-center fw-bold mb-3",
                            style={'color': COULEURS['navy_fonce'], 'fontSize': '3rem'}
                        ),
                        html.P(
                            "Probabilité de défaut",
                            className="text-center text-muted mb-3"
                        ),
                        dbc.Progress(
                            value=probabilite_defaut_lda*100,
                            color="danger" if probabilite_defaut_lda > 0.5 else "success",
                            className="mb-2",
                            style={'height': '20px'}
                        ),
                        html.P(
                            f"Classification : {'Défaut' if probabilite_defaut_lda > 0.5 else 'Non Défaut'}",
                            className=f"text-center fw-bold text-{'danger' if probabilite_defaut_lda > 0.5 else 'success'}"
                        )
                    ], className="py-4")
                ], className="shadow-sm border-0 h-100")
            ], xs=12, md=6, className="mb-4"),
            
            # Carte QDA
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        "QDA - Quadratic Discriminant Analysis",
                        className="text-white text-center fw-bold",
                        style={'backgroundColor': COULEURS['marron_fonce']}
                    ),
                    dbc.CardBody([
                        html.H2(
                            f"{probabilite_defaut_qda*100:.2f}%",
                            className="text-center fw-bold mb-3",
                            style={'color': COULEURS['marron_fonce'], 'fontSize': '3rem'}
                        ),
                        html.P(
                            "Probabilité de défaut",
                            className="text-center text-muted mb-3"
                        ),
                        dbc.Progress(
                            value=probabilite_defaut_qda*100,
                            color="danger" if probabilite_defaut_qda > 0.5 else "success",
                            className="mb-2",
                            style={'height': '20px'}
                        ),
                        html.P(
                            f"Classification : {'Défaut' if probabilite_defaut_qda > 0.5 else 'Non Défaut'}",
                            className=f"text-center fw-bold text-{'danger' if probabilite_defaut_qda > 0.5 else 'success'}"
                        )
                    ], className="py-4")
                ], className="shadow-sm border-0 h-100")
            ], xs=12, md=6, className="mb-4")
        ]),
        
        # Décision finale et recommandation
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.Div([
                        html.I(className=f"fas fa-{icone_decision} me-3", style={'fontSize': '2.5rem'}),
                        html.Div([
                            html.H3("Décision Finale", className="mb-2"),
                            html.H2(decision_finale, className="mb-3", style={'fontWeight': 'bold'}),
                        ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
                    ], className="text-center mb-3"),
                    
                    html.Hr(),
                    
                    html.Div([
                        html.P([
                            html.Strong("Modèle utilisé : "),
                            f"{nom_meilleur_modele} (F1-Score: {f1_meilleur_modele:.4f})"
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Probabilité de défaut selon {}: ".format(nom_meilleur_modele)),
                            html.Span(
                                f"{probabilite_recommandee*100:.2f}%",
                                style={'fontSize': '1.3rem', 'fontWeight': 'bold', 'color': couleur_modele_recommande}
                            )
                        ], className="mb-3"),
                        html.P([
                            html.Strong("Recommandation : "),
                            recommandation
                        ], className="mb-0")
                    ])
                    
                ], color=couleur_decision, className="border-0 shadow-sm")
            ], width=12)
        ])
    ])
    
    return resultats