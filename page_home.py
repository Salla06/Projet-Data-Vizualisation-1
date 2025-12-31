# ============================================================================
# PAGE D'ACCUEIL - TABLEAU DE BORD MICROFINANCE (VERSION DASH CORRIGÉE)
# ============================================================================
# Cette page sert de point d'entrée principal avec navigation élégante
# Structure :
# 1. Hero Section : Titre principal + sous-titre + badge année
# 2. Carousel : Images Getty de microfinance africaine
# 3. Navigation Grid : Cartes cliquables vers les sections du dashboard
# 4. Overview Section : Aperçu des domaines d'intervention
# 
# IMPORTANT : Placez le fichier home_styles.css dans le dossier assets/
# ============================================================================

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


# ----------------------------------------------------------------------------
# CONFIGURATION DES IMAGES DU CAROUSEL
# ----------------------------------------------------------------------------
# Images professionnelles de microfinance en Afrique

images_carousel = [
    {
        'src': 'https://media.gettyimages.com/id/912015114/fr/photo/people-walking-in-line-across-world-map-painted-on-asphalt-front-person-walking-left.jpg?s=612x612&w=0&k=20&c=At-5xME1xMGRQZno3zU6BBSd5z4-Ki_SQiskZ6xwAZE=',
        'header': 'Microfinance Inclusive',
        'caption': 'Accompagnement des entrepreneurs africains vers l\'autonomie financière'
    },
    {
        'src': 'https://media.gettyimages.com/id/2188078004/fr/photo/les-femmes-les-enfants-et-les-agriculteurs-commercialisent-en-plein-air-en-tant-que.jpg?s=612x612&w=0&k=20&c=byptpDX_u1UO9dCDUfOeKAn0Rg7nVlOTsEgVAPr8KwY=',
        'header': 'Autonomisation Économique',
        'caption': 'Solutions de financement adaptées aux besoins des communautés locales'
    },
    {
        'src': 'https://media.gettyimages.com/id/1345358368/fr/photo/portrait-large-dune-femme-africaine-travaillant-dans-un-magasin-de-march%C3%A9.jpg?s=612x612&w=0&k=20&c=aDLsgzcr0MaFDS-llJdHPZt63Nzk9FMF6qcPb3OB_wE=',
        'header': 'Commerce Local',
        'caption': 'Crédit accessible pour les petits commerçants et artisans'
    },
    {
        'src': 'https://media.gettyimages.com/id/517109675/fr/photo/red-chicory-processing.jpg?s=612x612&w=0&k=20&c=tnnMv_jaEKXMMNa_s3BjOVMvMhhaVvaH25uFdPUdzZc=',
        'header': 'Agriculture Durable',
        'caption': 'Financement des activités agricoles et développement rural'
    },
    {
        'src': 'https://media.gettyimages.com/id/1468520185/fr/photo/travail-d%C3%A9quipe-planification-et-r%C3%A9union-avec-les-m%C3%A9decins-et-lordinateur-portable-pour-les.jpg?s=612x612&w=0&k=20&c=zflF8eDoEgAHnThqKtsJWrZy9rlZZE7b6SydZsfPPGU=',
        'header': 'Développement Communautaire',
        'caption': 'Renforcement des capacités et inclusion financière'
    }
]


# ----------------------------------------------------------------------------
# CONSTRUCTION DU CAROUSEL AVEC COMPOSANT DASH BOOTSTRAP
# ----------------------------------------------------------------------------

carousel_items = []

for img in images_carousel:
    carousel_items.append({
        'key': img['header'],
        'src': img['src'],
        'img_style': {'height': '500px', 'objectFit': 'cover', 'width': '100%'},
        'caption_class_name': 'carousel-caption-custom',
        'header': img['header'],
        'caption': img['caption']
    })

carousel_component = dbc.Carousel(
    items=carousel_items,
    controls=True,
    indicators=True,
    interval=1000,
    className='custom-carousel'
)


# ----------------------------------------------------------------------------
# CONSTRUCTION DU LAYOUT DE LA PAGE D'ACCUEIL
# ----------------------------------------------------------------------------

home_layout = html.Div([
    
    # ========================================
    # HERO SECTION
    # ========================================
    
    html.Div([
        dbc.Col([
            html.Div([
                html.H1("Tableau de Bord Microfinance", className="header-title"),
                html.P("Analyse de Risque Crédit et Gestion de Portefeuille", className="header-subtitle")
            ]),
        
        ], className="header-content")
    ], className="main-header"),
    
    # ========================================
    # CONTENU PRINCIPAL
    # ========================================
    
    html.Div([
        
        # ========================================
        # CAROUSEL
        # ========================================
        
        html.Div([
            carousel_component
        ], className="carousel-wrapper"),
        
        # ========================================
        # NAVIGATION GRID
        # ========================================
        
        html.Div([
            
            # Carte 1 : Analyse & Exploration
            dcc.Link([
                html.Div([
                    html.I(className="fas fa-chart-line")
                ], className="nav-icon"),
                html.H3("Analyse & Exploration", className="nav-title"),
                html.P(
                    "Explorez les données de crédit, analysez les tendances et identifiez les facteurs de risque clés",
                    className="nav-description"
                ),
                html.Div([
                    html.Span("Voir plus"),
                    html.I(className="fas fa-arrow-right", style={'marginLeft': '0.5rem'})
                ], className="nav-link-text")
            ], href="/analysis", className="nav-card"),
            
            # Carte 2 : Modélisation
            dcc.Link([
                html.Div([
                    html.I(className="fas fa-brain")
                ], className="nav-icon"),
                html.H3("Modélisation Prédictive", className="nav-title"),
                html.P(
                    "Utilisez les modèles LDA et QDA pour prédire le risque de défaut et optimiser les décisions de crédit",
                    className="nav-description"
                ),
                html.Div([
                    html.Span("Voir plus"),
                    html.I(className="fas fa-arrow-right", style={'marginLeft': '0.5rem'})
                ], className="nav-link-text")
            ], href="/modeling", className="nav-card")
            
        ], className="nav-grid"),
    ], style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'paddingBottom': '3rem'})
    
])


# ============================================================================
# FIN DU FICHIER
# ============================================================================