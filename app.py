import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Startups - Dashboard Professionnel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    .stPlotlyChart {width: 100% !important;}
    h1 {font-weight: 700; margin-bottom: 0.5rem;}
    h2 {font-weight: 600; margin-top: 2rem;}
    .metric-container {background-color: #f8f9fa; padding: 1rem; border-radius: 8px;}
 
    :root {
    --bg-color: #C29C8F;      /* bleu ciel clair */
    --text-dark: #1E3A5F;    /* bleu foncé lisible */
}

/* GLOBAL RESET */
.stApp {
    background-color: var(--bg-color);
    font-family: 'Inter', sans-serif;
    color: var(--text-dark);
}

/* Headings */
h1, h2 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
}

    </style>
    """, unsafe_allow_html=True)

# Chargement des données avec cache
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("50_Startups.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Définition des colonnes numériques (une seule fois, au niveau global)
numeric_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']

# Sidebar - Navigation
st.sidebar.title("Navigation")
pages = [
    "Vue d'ensemble",
    "Analyse Exploratoire",
    "Corrélations",
    "Analyse par État",
    "Modélisation Prédictive",
    "Prédicteur Interactif"
]
page = st.sidebar.radio("Sélectionner une section", pages)

# ========================
# 1. Vue d'ensemble
# ========================
if page == "Vue d'ensemble":
    st.title("Analyse des Performances de 50 Startups")
    st.markdown("### Dataset contenant les dépenses en R&D, Administration, Marketing, l'État d'implantation et le Profit annuel.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de Startups", len(df))
    with col2:
        st.metric("Profit Moyen", f"${df['Profit'].mean():,.0f}")
    with col3:
        st.metric("Dépenses R&D Moyennes", f"${df['R&D Spend'].mean():,.0f}")
    with col4:
        st.metric("Dépenses Marketing Moyennes", f"${df['Marketing Spend'].mean():,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.subheader("Aperçu du Dataset")
        st.dataframe(df.style.format({
            'R&D Spend': '${:,.2f}',
            'Administration': '${:,.2f}',
            'Marketing Spend': '${:,.2f}',
            'Profit': '${:,.2f}'
        }), use_container_width=True)
    with col_right:
        st.subheader("Structure des Données")
        st.write("Lignes × Colonnes :", df.shape)
        st.write("Types de données :")
        st.dataframe(df.dtypes.rename("Type").to_frame())

# ========================
# 2. Analyse Exploratoire
# ========================
elif page == "Analyse Exploratoire":
    st.header("Analyse Exploratoire des Données")

    selected_var = st.selectbox("Sélectionner une variable numérique", numeric_cols)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x=selected_var, color="State",
            nbins=20, marginal="box",
            title=f"Distribution de {selected_var}",
            labels={selected_var: selected_var.replace(' ', '<br>')}
        )
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Statistiques descriptives - {selected_var}")
        desc = df[selected_var].describe().to_frame().T
        st.dataframe(desc.style.format("{:,.2f}"))

        st.markdown("**Top 5 startups par cette variable**")
        top5 = df.nlargest(5, selected_var)[['State', selected_var, 'Profit']].copy()
        top5[selected_var] = top5[selected_var].map('${:,.2f}'.format)
        top5['Profit'] = top5['Profit'].map('${:,.2f}'.format)
        st.table(top5)

    st.markdown("---")
    st.subheader("Matrice de dispersion des variables numériques")
    fig_pair = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color="State",
        title="Relations bivariées entre variables numériques",
        height=800
    )
    st.plotly_chart(fig_pair, use_container_width=True)

# ========================
# 3. Corrélations
# ========================
elif page == "Corrélations":
    st.header("Analyse des Corrélations")

    corr_matrix = df[numeric_cols].corr()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Matrice de Corrélation")
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            hovertemplate='%{x} vs %{y}: <b>%{text}</b><extra></extra>'
        ))
        fig_heatmap.update_layout(
            height=500,
            title="Corrélation de Pearson"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.subheader("Relation avec le Profit")
        x_var = st.selectbox("Variable en abscisse", [col for col in numeric_cols if col != 'Profit'])
        
        fig_scatter = px.scatter(
            df, x=x_var, y='Profit', color="State",
            size='Administration', hover_data=['State'],
            trendline="ols",
            title=f"{x_var} vs Profit (avec régression linéaire)",
            labels={'Profit': 'Profit ($)', x_var: x_var}
        )
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.info("La dépense en R&D présente la plus forte corrélation positive avec le Profit (typiquement > 0.95).")

# ========================
# 4. Analyse par État
# ========================
elif page == "Analyse par État":
    st.header("Analyse Comparative par État")

    summary = df.groupby('State').agg({
        'Profit': ['mean', 'median', 'std', 'count'],
        'R&D Spend': 'mean',
        'Marketing Spend': 'mean',
        'Administration': 'mean'
    }).round(2)

    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            summary, x='State', y='Profit_mean',
            title="Profit Moyen par État",
            text='Profit_mean',
            labels={'Profit_mean': 'Profit Moyen ($)'}
        )
        fig_bar.update_traces(texttemplate='$%{text:,.0f}')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_box = px.box(
            df, x='State', y='Profit', color='State',
            title="Distribution du Profit par État"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("Tableau Récapitulatif par État")
    display_summary = summary.copy()
    for col in display_summary.columns[1:]:
        if 'Profit' in col or 'Spend' in col:
            display_summary[col] = display_summary[col].map('${:,.2f}'.format)
    st.dataframe(display_summary, use_container_width=True)

# ========================
# 5. Modélisation Prédictive
# ========================
elif page == "Modélisation Prédictive":
    st.header("Modélisation du Profit")

    # Préparation des données
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=['State'], drop_first=True)

    X = df_model.drop('Profit', axis=1)
    y = df_model['Profit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèles
    models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        results[name] = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }

    results_df = pd.DataFrame(results).T.round(4)
    st.subheader("Performance des Modèles")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#d4edda'))

    # Importance des variables (Random Forest)
    rf_model = models["Random Forest"]
    importances = pd.DataFrame({
        'Variable': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_imp = px.bar(
        importances, x='Importance', y='Variable',
        orientation='h', title="Importance des Variables (Random Forest)"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Prédictions vs Réel
    fig_pred = px.scatter(
        x=y_test, y=predictions["Random Forest"],
        labels={'x': 'Profit Réel', 'y': 'Profit Prédit'},
        title="Prédictions vs Valeurs Réelles (Random Forest)"
    )
    min_val = min(y_test.min(), predictions["Random Forest"].min())
    max_val = max(y_test.max(), predictions["Random Forest"].max())
    fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                       line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_pred, use_container_width=True)

# ========================
# 6. Prédicteur Interactif
# ========================
else:
    st.header("Prédicteur de Profit")

    st.markdown("Saisir les valeurs des dépenses et sélectionner l'État pour estimer le profit annuel.")

    col1, col2 = st.columns(2)
    with col1:
        rd = st.slider("Dépenses R&D ($)", 
                       float(df['R&D Spend'].min()), 
                       float(df['R&D Spend'].max()), 
                       float(df['R&D Spend'].median()),
                       step=1000.0)
        admin = st.slider("Dépenses Administration ($)", 
                          float(df['Administration'].min()), 
                          float(df['Administration'].max()), 
                          float(df['Administration'].median()),
                          step=1000.0)
    with col2:
        marketing = st.slider("Dépenses Marketing ($)", 
                              float(df['Marketing Spend'].min()), 
                              float(df['Marketing Spend'].max()), 
                              float(df['Marketing Spend'].median()),
                              step=1000.0)
        state = st.selectbox("État", options=sorted(df['State'].unique()))

    # Préparation des features
    input_df = pd.DataFrame({
        'R&D Spend': [rd],
        'Administration': [admin],
        'Marketing Spend': [marketing],
        'State_California': [1 if state == 'California' else 0],
        'State_Florida': [1 if state == 'Florida' else 0],
        'State_New York': [1 if state == 'New York' else 0]
    })

    model_columns = ['R&D Spend', 'Administration', 'Marketing Spend',
                     'State_California', 'State_Florida', 'State_New York']
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Entraînement sur l'ensemble des données pour meilleure précision
    full_X = pd.get_dummies(df, columns=['State'], drop_first=True)
    full_y = full_X['Profit']
    full_X = full_X.drop('Profit', axis=1)

    lin_model = LinearRegression().fit(full_X.reindex(columns=model_columns, fill_value=0), full_y)
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42).fit(full_X.reindex(columns=model_columns, fill_value=0), full_y)

    pred_lin = lin_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]
    pred_avg = (pred_lin + pred_rf) / 2

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Régression Linéaire", f"${pred_lin:,.0f}")
    with col_b:
        st.metric("Random Forest", f"${pred_rf:,.0f}")
    with col_c:
        st.metric("Moyenne des Modèles", f"${pred_avg:,.0f}")

    st.success("La dépense en R&D est le principal déterminant du profit, suivie par les dépenses marketing.")