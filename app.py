# ============================================================================
# APPLICATION PRINCIPALE - DASHBOARD MICROFINANCE
# ============================================================================
# Ce fichier configure l'application Dash avec navigation multi-pages,
# thème personnalisé et charte graphique institutionnelle
# 
# IMPORTANT : Les styles de la navbar sont répartis comme suit :
# - Styles de base (couleurs, dimensions) : définis dans ce fichier Python
# - Styles interactifs (hover, active) : définis dans assets/navbar_styles.css
# ============================================================================

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


# ----------------------------------------------------------------------------
# INITIALISATION DE L'APPLICATION DASH
# ----------------------------------------------------------------------------
# Configuration avec Bootstrap et Font Awesome pour les icônes

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v6.1.1/css/all.css"
    ],
    suppress_callback_exceptions=True,
    title="Dashboard Microfinance | Analyse de Risque Crédit"
)

server = app.server  # Pour le déploiement en production


# ----------------------------------------------------------------------------
# DÉFINITION DE LA PALETTE DE COULEURS INSTITUTIONNELLE
# ----------------------------------------------------------------------------
# Inspirée des couleurs naturelles du baobab et de la terre africaine

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
# STYLES CSS DE BASE POUR LA NAVBAR
# ----------------------------------------------------------------------------
# Ces styles définissent l'apparence de base de la navbar
# Les effets interactifs (hover, active) sont gérés par navbar_styles.css

styles_navbar = {
    'container_navbar': {
        'backgroundColor': COULEURS['navy_fonce'],
        'boxShadow': '0 2px 8px rgba(0,0,0,0)',
        'padding': '0.4rem 0',
        'borderBottom': '1px solid rgba(255,255,255,0.1)'
    },
    'logo_container': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px'
    },
    'logo_image': {
        'height': '38px',
        'width': '38px',
        'objectFit': 'contain'
    },
    'brand_text': {
        'color': COULEURS['blanc'],
        'fontSize': '1.2rem',
        'fontWeight': '600',
        'margin': '0',
        'letterSpacing': '0.3px',
        'lineHeight': '1.2'
    },
    'brand_subtitle': {
        'color': COULEURS['dore'],
        'fontSize': '0.9rem',
        'margin': '0',
        'textTransform': 'uppercase',
        'letterSpacing': '1.5px',
        'fontWeight': '400',
        'lineHeight': '1.2',
        'marginTop': '2px'
    },
    'nav_link_base': {
        'color': COULEURS['blanc'],
        'fontSize': '1.1rem',
        'fontWeight': '400',
        'padding': '0.5rem 1rem',
        'margin': '0 0.3rem',
        'borderRadius': '0'
    }
}


# ----------------------------------------------------------------------------
# CONSTRUCTION DE LA BARRE DE NAVIGATION
# ----------------------------------------------------------------------------
# Header professionnel avec logo, titre et menu de navigation - Style PUDC
# Note : Les effets hover et active sont gérés par assets/navbar_styles.css

navbar = dbc.Navbar(
    dbc.Container([
        # Section gauche : Logo et titre
        dbc.Row([
            dbc.Col([
                html.A(
                    html.Div([
                        html.Img(
                            src='/assets/logo.png',
                            style=styles_navbar['logo_image']
                        ),
                        html.Div([
                            html.H1(
                                "Dashboard Microfinance",
                                style=styles_navbar['brand_text']
                            ),
                            html.P(
                                "Analyse de Risque Crédit",
                                style=styles_navbar['brand_subtitle']
                            )
                        ])
                    ], style=styles_navbar['logo_container']),
                    href="/",
                    style={'textDecoration': 'none'}
                )
            ], width='auto')
        ], align="center"),
        
        # Section droite : Menu de navigation
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(
                        dbc.NavLink(
                             "Accueil",
                            href="/",
                            active="exact",
                            style=styles_navbar['nav_link_base']
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "Analyse & Exploration",
                            href="/analysis",
                            active="exact",
                            style=styles_navbar['nav_link_base']
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "Modélisation",
                            href="/modeling",
                            active="exact",
                            style=styles_navbar['nav_link_base']
                        )
                    )
                ], navbar=True)
            ], width='auto')
        ], align="center", className="ms-auto")
        
    ], fluid=True, className="px-4"),
    dark=True,
    color="dark",
    style=styles_navbar['container_navbar'],
    className="mb-0"
)


# ----------------------------------------------------------------------------
# DÉFINITION DU LAYOUT PRINCIPAL
# ----------------------------------------------------------------------------
# Structure avec routing pour navigation multi-pages

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', style={'minHeight': '100vh'})
], style={'backgroundColor': COULEURS['gris_clair']})


# ----------------------------------------------------------------------------
# IMPORTATION DES CALLBACKS
# ----------------------------------------------------------------------------
# Enregistrement des callbacks pour interactivité

import callbacks


# ----------------------------------------------------------------------------
# LANCEMENT DU SERVEUR
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)


# ============================================================================
# FIN DU FICHIER
# ============================================================================