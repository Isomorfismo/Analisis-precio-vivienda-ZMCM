import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import numpy as np
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import unidecode

# Cargar datos
df_limpio = pd.read_csv('df_venta_limpio.csv')
df_normalizado = pd.read_csv('df_venta_normalizado.csv')
df_combined = gpd.read_file('df_combined/df_combined.shp')

df_combined = df_combined.drop_duplicates(subset=['OBJECTID'])
# Datos de servicios
servicios_shape = gpd.read_file('servicios.shp')
servicios_shape = servicios_shape.drop_duplicates(subset=['OBJECTID'])

precio_promedio = df_limpio['Precio_MXN'].mean()
precio_promedio_m2 = df_combined['precio_m2'].mean()
total_viviendas = len(df_limpio)

# Mapa de precios con escala perceptual y estilo urbano
fig_map = px.choropleth_mapbox(
    df_combined,
    geojson=df_combined.geometry,
    locations=df_combined.index,
    color='precio_m2',
    color_continuous_scale="Turbo",
    mapbox_style="carto-positron",
    zoom=11,
    center={"lat": 19.4326, "lon": -99.1332},
    opacity=0.7,
    labels={'precio_m2': 'Precio m² (MXN)'},
    title='Precio m² en CDMX y Edomex',
    hover_data={'MUN_NAME': True, 'SETT_NAME': True, 'precio_m2': ':.2f'}
)
fig_map.update_layout(font=dict(size=14), title='Precio m² en CDMX y Edomex', autosize=True)
df_limpio['Estado'] = df_limpio['Estado'].replace({'DISTRITO FEDERAL': 'Ciudad de México', 'MEXICO': 'Estado de México'})

df_filtered = df_limpio[df_limpio['Estado'].isin(['Ciudad de México', 'Estado de México'])]
# Proporcion Casas/departamentos
tot_estados = df_filtered.groupby("Estado").size().reset_index(name="cantidad")

proporcion_fig = go.Figure(data=[go.Pie(
    labels=tot_estados["Estado"],
    values=tot_estados["cantidad"],
    textinfo='label+percent',
    hole=0.4,
    showlegend=False
)])

proporcion_fig.update_layout(height=150, width=300, margin={"r":0,"t":0,"l":0,"b":0})

# Calcular el promedio de precios por colonia
colonias_promedio = df_combined.groupby('SETT_NAME').agg(
    INDEX=('precio_m2', 'mean')
).reset_index().rename(columns={'SETT_NAME': 'COLONIA'})

# Seleccionar las 10 colonias más caras y ordenar de mayor a menor
top_colonias_df = colonias_promedio.sort_values(by='INDEX', ascending=False).head(10)

bar_top_colonias = px.bar(
    top_colonias_df.sort_values('INDEX', ascending=True),  # Para que la barra más alta quede arriba
    x='INDEX',
    y='COLONIA',
    title='Top 10 Colonias Más Caras (Precio Promedio INDEX)',
    labels={'COLONIA': 'Colonia', 'INDEX': '(MXN/m²)'},
    color='INDEX',
    color_continuous_scale='Reds',
    orientation='h'
)
bar_top_colonias.update_traces(marker_line_width=2, marker_line_color='black')
bar_top_colonias.update_layout(xaxis_tickangle=0, width=700, font=dict(size=10))

# Calcular la cantidad de viviendas por número de baños, recámaras y estacionamientos
superficie_counts = df_limpio['Superficie'].value_counts().reset_index()
superficie_counts.columns = ['Superficie', 'Cantidad de Viviendas']
superficie_counts = superficie_counts.sort_values('Superficie')

fig_caracteristicas = go.Figure()
fig_caracteristicas.add_trace(go.Histogram(x=df_limpio['Superficie'], nbinsx=30, marker_color='#00CC96'))
fig_caracteristicas.update_layout(showlegend=False, height=500, font=dict(size=10), width=800, title='Superficie promedio de viviendas')
#agregarle contornos a las barras
fig_caracteristicas.update_traces(marker_line_width=2, marker_line_color='black')
# Inicializar Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

#============================================================
# PÁGINA DE INICIO
#============================================================

pagina_inicio = html.Div([
    html.Section([
        html.Div(id="kpi-section", children=[
            html.Div(id="kpi-precio-promedio", children=[
                html.P("Precio promedio"),
                html.H5(f"$ {precio_promedio:,.2f}", className="card-title")
            ]),
            html.Div(id="kpi-precio-promedio-m2", children=[
                html.P("Precio por metro cuadrado"),
                html.H5(f"$ {precio_promedio_m2:,.2f}", className="card-title")
            ]),
            html.Div(id="kpi-total-viviendas", children=[
                html.P("Total de viviendas"),
                html.H5(f"{total_viviendas:,}", className="card-title")
            ]),
            html.Div(id="kpi-proporcion-fig", children=[
                html.P("Distribución de viviendas por estado"),
                dcc.Graph(figure=proporcion_fig, id="proporcion-fig")
            ])
        ])
    ]),
    html.Section([
        html.Div([
            dcc.Graph(figure=fig_map, id="map-fig")
        ]),
        html.Div([
            dcc.Dropdown(
                id='caracteristica-dropdown',
                options=[
                    {'label': 'Superficie', 'value': 'Superficie'},
                    {'label': 'Baños', 'value': 'Baños'},
                    {'label': 'Recámaras', 'value': 'Recamaras'},
                    {'label': 'Estacionamientos', 'value': 'Estacionamientos'},
                    {'label': 'Balcón', 'value': 'Balcon'},
                    {'label': 'Terraza', 'value': 'Terraza'},
                    {'label': 'Jardín', 'value': 'Jardin'},
                    {'label': 'Piscina', 'value': 'Piscina'},
                    {'label': 'Seguridad', 'value': 'Seguridad'},
                    {'label': 'Tipo', 'value': 'Tipo'},
                ],
                value='Superficie'
            ),
            dcc.Graph(figure=fig_caracteristicas, id="triple-fig")
        ]),
        html.Div([
            dcc.Graph(figure=bar_top_colonias, id="top-colonias-fig")
        ])
    ], id="graphs-section-main")
])


#============================================================
# PÁGINA DE SERVICIOS Y ENTORNO
#============================================================
fig_map_servicios = px.choropleth_mapbox(
    servicios_shape,
    geojson=servicios_shape.geometry,
    locations=servicios_shape.index,
    color='Hospitales',
    color_continuous_scale="Turbo",
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 19.4326, "lon": -99.1332},
    opacity=0.7,
    title='Servicios en CDMX y Edomex',
    hover_data={'MUN_NAME': True, 'SETT_NAME': True, 'ST_NAME': True}
)
top_colonias_servicios = servicios_shape.sort_values(by='Hospitales', ascending=False).head(10)

fig_top_servicios = px.bar(
    top_colonias_servicios.sort_values('Hospitales', ascending=True),
    y='SETT_NAME',
    title='Top 10 Colonias con más Hospitales',
    labels={'SETT_NAME': 'Colonia', 'Hospitales': 'No. de Hospitales'},
    color='Hospitales',
    color_continuous_scale='viridis',
    orientation='h'
)

pagina_servicios = html.Div([
    html.Div([
        dcc.Dropdown(
        id='servicio-dropdown',
        options=[
            {'label': 'Hospitales', 'value': 'Hospitales'},
            {'label': 'Escuelas', 'value': 'Escuelas'},
            {'label': 'Transporte Público', 'value': 'Transporte'},
            {'label': 'Restaurantes', 'value': 'Restaurant'},
            {'label': 'Carpetas', 'value': 'Carpetas'},
            {'label': 'Esparcimiento', 'value': 'Esparcimie'}
        ],
        value='Hospitales'),
        dcc.Graph(figure=fig_map_servicios,id="map-servicios-fig", style={"height": "600px"})
    ] , id="servicios-section-dropdown-map"),
    html.Div([
        dcc.Graph(figure=fig_top_servicios, id="top-colonias-servicios-fig", style={"height": "600px"})
    ])
], id="servicios-section")

#============================================================
# PÁGINA DE MODELOS
#============================================================

# modelo_xgb = joblib.load('C:/Users/yahir/Documents/Python/TT/Models/best_model_XGBoost (1).joblib')

pagina_modelos = html.Div([
    html.Div([
        html.Div([
            html.Label('Superficie (m²):'),
            dcc.Input(id='input-superficie', type='number', min=1, step=1, value=60),
            html.Label('Baños:'),
            dcc.Input(id='input-banos', type='number', min=1, step=1, value=1),
            html.Label('Recámaras:'),
            dcc.Input(id='input-recamaras', type='number', min=1, step=1, value=2),
            html.Label('Estacionamientos:'),
            dcc.Input(id='input-estacionamientos', type='number', min=0, step=1, value=1),
            html.Label('Balcón:'),
            dcc.Input(id='input-balcon', type='number', min=0, step=1, value=0),
            html.Label('Terraza:'),
            dcc.Input(id='input-terraza', type='number', min=0, step=1, value=0),
            html.Label('Jardín:'),
            dcc.Input(id='input-jardin', type='number', min=0, step=1, value=0),
            html.Label('Piscina:'),
            dcc.Input(id='input-piscina', type='number', min=0, step=1, value=0),
            html.Label('Seguridad:'),
            dcc.Input(id='input-seguridad', type='number', min=0, step=1, value=0),
            html.Label('Tipo (Casa/Depto):'),
            dcc.Input(id='input-tipo', type='text', value='Casa'),
            html.Label('Colonia:'),
            dcc.Input(id='input-colonia', type='text', value='DEL GAS'),
            html.Label('Municipio:'),
            dcc.Input(id='input-municipio', type='text', value='AZCAPOTZALCO'),
            html.Label('Estado:'),
            dcc.Input(id='input-estado', type='text', value='DISTRITO FEDERAL')
        ]),
        html.Div([
            html.Button('Predecir Precio', id='btn-predecir', n_clicks=0)
        ]),
    ], id='input-form-div'),
    html.Div([
        html.H3('Resultado de la Predicción:', style={'marginTop': '20px'}),
        html.H4(id='output-prediccion', style={'marginTop': '30px', 'fontSize': '22px', 'color': 'blue'}),
    ])

], id='modelos-section')


#============================================================
# PÁGINA DE COLONIAS
#============================================================
pagina_colonias = html.Div([
    html.Div([
        dcc.Graph(id='colonia-map-fig'),
    ], id='colonia-map-div'),
    html.Div([
        html.H2(id='colonia-name'),
        html.Div(id='colonia-stats')
    ], id='colonia-info-div')
], id='colonia-section')

app.layout = html.Main([
    #===========================================================
    # LAYOUT PRINCIPAL DEL DASHBOARD
    #===========================================================
    html.Nav([
        dcc.Dropdown(
            id='colonia-municipio-dropdown',
            options=[
                {
                    'label': f"{row['SETT_NAME']}, {row['MUN_NAME']}, {row['ST_NAME']}",
                    'value': f"{row['SETT_NAME']}|{row['MUN_NAME']}|{row['ST_NAME']}"
                }
                for _, row in df_combined.drop_duplicates(subset=['SETT_NAME', 'MUN_NAME', 'ST_NAME']).iterrows()
            ],
            placeholder='Selecciona colonia y municipio',
            value=None
        ),
        html.Div([
            html.Div(dcc.Link('Inicio', href='/', style={'margin-right': '20px'})),
            html.Div(dcc.Link('Servicios y entorno', href='/serviciosentorno', style={'margin-right': '20px'})),
            html.Div(dcc.Link('Predicción', href='/modelos', style={'margin-right': '20px'})),
        ], id='nav-links'),
    ], className='navbar'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
@app.callback(
    Output('top-colonias-servicios-fig', 'figure'),
    Output('map-servicios-fig', 'figure'),
    Input('servicio-dropdown', 'value')
)
def update_servicios_figures(selected_service):
    top_colonias_servicios = servicios_shape.sort_values(by=selected_service, ascending=False).head(10)
    fig_top_servicios = px.bar(
        top_colonias_servicios.sort_values(selected_service, ascending=True),
        x=selected_service,
        y='SETT_NAME',
        title=f'Top 10 Colonias con más {selected_service}',
        labels={'SETT_NAME': 'Colonia', selected_service: f'No. de {selected_service}'},
        color=selected_service,
        color_continuous_scale='viridis',
        orientation='h'
    )
    fig_map_servicios = px.choropleth_mapbox(
        servicios_shape,
        geojson=servicios_shape.geometry,
        locations=servicios_shape.index,
        color=selected_service,
        color_continuous_scale="Turbo",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 19.4326, "lon": -99.1332},
        opacity=0.7,
        title='Servicios en CDMX y Edomex',
        hover_data={'MUN_NAME': True, 'SETT_NAME': True, 'ST_NAME': True}
    )
    fig_map_servicios.update_layout(height=600)
    fig_top_servicios.update_layout(height=600)
    return fig_top_servicios, fig_map_servicios

def criminalidad(carpetas):
    rango = {
        "Muy baja": (10, 249),
        "Baja": (250, 749),
        "Media": (750, 1499),
        "Alta": (1500, 2999),
        "Muy alta": (3000, float("inf"))
    }
    return next((key for key, (low, high) in rango.items() if low <= carpetas <= high), "Desconocido")


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('colonia-municipio-dropdown', 'value')
)
def display_page(pathname, colonia_municipio):
    if colonia_municipio:
        colonia, municipio, estado = colonia_municipio.split('|')
        datos_colonia_map = df_combined[
            (df_combined['SETT_NAME'] == colonia) &
            (df_combined['MUN_NAME'] == municipio) &
            (df_combined['ST_NAME'] == estado)
        ]
        datos_colonia_info = df_normalizado[
            (df_normalizado['Colonia'] == colonia) &
            (df_normalizado['Municipio'] == municipio) &
            (df_normalizado['Estado'] == estado)
        ]
        if datos_colonia_map.empty or datos_colonia_map['precio_m2'].isnull().all():
            fig_empty = go.Figure()
            fig_empty.update_layout(
                title="No hay datos para la colonia seleccionada",
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[{
                    'text': "No hay datos disponibles",
                    'xref': "paper",
                    'yref': "paper",
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return html.Div([
                dcc.Graph(figure=fig_empty, id='colonia-map-div'),
                html.Div([
                    html.H1(f"COLONIA {colonia}, {municipio}"),
                    html.Div([
                        html.P("No hay datos disponibles para esta colonia.")
                    ], id='colonia-stats')
                ], id='colonia-info-div')
            ], id='colonia-section')
        else:
            fig_map_colonia = px.choropleth_mapbox(
                df_combined,
                geojson=df_combined.geometry,
                locations=df_combined.index,
                color='precio_m2',
                color_continuous_scale="Turbo",
                mapbox_style="carto-positron",
                zoom=14,
                center={"lat": datos_colonia_map.geometry.centroid.y.mean(), "lon": datos_colonia_map.geometry.centroid.x.mean()},
                opacity=0.5,
                labels={'precio_m2': 'Precio m² (MXN)'},
                hover_data={'MUN_NAME': True, 'SETT_NAME': True, 'precio_m2': ':.2f'}
            ).update_layout(height=600)
            return html.Div([
                dcc.Graph(figure=fig_map_colonia, id='colonia-map-div'),
                html.Div([
                    html.H1(f"COLONIA {colonia}, {municipio}"),
                    html.Div([
                        html.H3("Estadísticas de la colonia"),
                        html.P(f"Precio promedio m²: ${datos_colonia_map['precio_m2'].mean():,.2f} MXN"),
                        html.P(f"Tamaño promedio de vivienda: {datos_colonia_info['Superficie'].mean():.2f} m²"),
                        html.P(f"Número promedio de baños: {datos_colonia_info['Baños'].mean():.0f}"),
                        html.P(f"Número promedio de recámaras: {datos_colonia_info['Recamaras'].mean():.0f}")
                    ], id='colonia-stats', style={'marginTop': '1rem', 'backgroundColor': '#f8f8f8', 'padding': '0.8rem', 'borderRadius': '0.8rem'}),
                    html.Div([
                        html.H3("Servicios en la colonia"),
                        html.P(f"Hospitales: {datos_colonia_info['Hospitales'].mean():.0f}"),
                        html.P(f"Escuelas: {datos_colonia_info['Escuelas'].mean():.0f}"),
                        html.P(f"Esparcimiento: {datos_colonia_info['Esparcimiento'].mean():.0f}"),
                        html.P(f"Restaurantes: {datos_colonia_info['Restaurantes'].mean():.0f}"),
                    ], id='colonia-servicios-div', style={'marginTop': '1rem', 'backgroundColor': '#f8f8f8', 'padding': '0.8rem', 'borderRadius': '0.8rem'}),
                    html.Div([
                        html.H3("Incidencia Criminal"),
                        (lambda cat: html.P([
                            "Criminalidad: ",
                            html.Span(cat, style={
                                "color": {
                                    "Muy baja": "green",
                                    "Baja": "limegreen",
                                    "Media": "orange",
                                    "Alta": "red",
                                    "Muy alta": "purple"
                                }.get(cat, "black")
                            })
                        ]))(criminalidad(datos_colonia_info['Carpetas'].mean())),
                        html.P(f"Estaciones de transporte público: {datos_colonia_info['Transporte'].mean():.0f}")
                    ], id='colonia-info-div', style={'marginTop': '1rem', 'backgroundColor': '#f8f8f8', 'padding': '0.8rem', 'borderRadius': '0.8rem', 'margin-right': '0'})
                ], id='colonia-info-div')
            ], id='colonia-section')
    elif pathname == '/serviciosentorno':
        return pagina_servicios
    elif pathname == '/modelos':
        return pagina_modelos
    else:
        return pagina_inicio

@app.callback(
    Output('triple-fig', 'figure'),
    Input('caracteristica-dropdown', 'value')
)
def update_triple_figure(selected_characteristic):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_limpio[selected_characteristic],
        nbinsx=30
    ))
    fig.update_layout(
        showlegend=False,
        height=500,
        font=dict(size=10),
        width=600,
        title=f'{selected_characteristic} promedio de viviendas'
    )
    fig.update_traces(marker_line_width=2, marker_line_color='black')
    return fig

def preprocess_and_predict(new_property, best_xgboost_model):

    # =============================
    # 1. Asegurar columnas booleanas
    # =============================
    bool_cols_input = [c for c in ['Seguridad','Balcon','Piscina','Jardin','Terraza'] 
                       if c in new_property.columns]
    for c in bool_cols_input:
        new_property[c] = new_property[c].astype('Float64')

    # =============================
    # 2. Cargar dataset original
    # =============================
    original_full_df = pd.read_csv('df_precios_con_servicios.csv', low_memory=False)

    # ===== Normalizar texto =====
    def norm_text(s):
        if pd.isna(s):
            return ""
        s = str(s).strip().upper()
        s = unidecode.unidecode(s)
        return " ".join(s.split())

    for col in ['Colonia','Municipio','Estado']:
        if col in original_full_df.columns:
            original_full_df[col] = original_full_df[col].apply(norm_text)
        else:
            original_full_df[col] = ""

    for col in ['Colonia','Municipio','Estado']:
        new_property[col] = new_property[col].apply(norm_text)

    def map_estado_cdmx_to_df(estado):
        if not estado:
            return estado
        s = estado.replace(".", "").strip()
        if "CIUDAD" in s and "MEX" in s:
            return "DISTRITO FEDERAL"
        return s

    original_full_df['Estado'] = original_full_df['Estado'].apply(map_estado_cdmx_to_df)
    new_property['Estado'] = new_property['Estado'].apply(map_estado_cdmx_to_df)

    # Crear columnas Upper para consistencia
    original_full_df['Colonia_Upper'] = original_full_df['Colonia']
    original_full_df['Municipio_Upper'] = original_full_df['Municipio']
    original_full_df['Estado_Upper'] = original_full_df['Estado']

    # =============================
    # 3. Obtener valores de servicios
    # =============================
    service_cols = ['Hospitales','Escuelas','Esparcimiento',
                    'Restaurantes','Carpetas','Transporte']

    search_colonia = new_property['Colonia'].iloc[0]
    search_municipio = new_property['Municipio'].iloc[0]
    search_estado = new_property['Estado'].iloc[0]

    fetched_service_values = {col: np.nan for col in service_cols}

    # 1) Coincidencia exacta
    exact_match = original_full_df[
        (original_full_df['Colonia_Upper'] == search_colonia) &
        (original_full_df['Municipio_Upper'] == search_municipio) &
        (original_full_df['Estado_Upper'] == search_estado)
    ]

    if not exact_match.empty:
        for col in service_cols:
            vals = exact_match[col].dropna()
            if not vals.empty:
                fetched_service_values[col] = vals.iloc[0]

    # 2) Media por Municipio + Estado
    missing = [c for c,v in fetched_service_values.items() if pd.isna(v)]
    if missing:
        mun_match = original_full_df[
            (original_full_df['Municipio_Upper'] == search_municipio) &
            (original_full_df['Estado_Upper'] == search_estado)
        ]
        if not mun_match.empty:
            for col in missing:
                mean_val = mun_match[col].mean()
                if not pd.isna(mean_val):
                    fetched_service_values[col] = mean_val

    # 3) Mediana global
    still_missing = [c for c,v in fetched_service_values.items() if pd.isna(v)]
    for col in still_missing:
        median_val = original_full_df[col].median()
        fetched_service_values[col] = median_val if not pd.isna(median_val) else 0.0

    # Agregar servicios al input
    for col,val in fetched_service_values.items():
        new_property[col] = val

    # =============================
    # 4. Predicción
    # =============================
    final_df = new_property.copy()
    pred_log = best_xgboost_model.predict(final_df)
    pred_price = np.exp(pred_log)[0]

    # SOLO DEVUELVES LA PREDICCIÓN
    return float(pred_price)

best_xgboost_model = joblib.load('best_model_XGBoost.joblib')

@app.callback(
    Output('output-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    State('input-superficie', 'value'),
    State('input-banos', 'value'),
    State('input-recamaras', 'value'),
    State('input-estacionamientos', 'value'),
    State('input-balcon', 'value'),
    State('input-terraza', 'value'),
    State('input-jardin', 'value'),
    State('input-piscina', 'value'),
    State('input-seguridad', 'value'),
    State('input-tipo', 'value'),
    State('input-colonia', 'value'),
    State('input-municipio', 'value'),
    State('input-estado', 'value')
)
def predecir_precio(n_clicks, superficie, banos, recamaras, estacionamientos, balcon, terraza, jardin, piscina, seguridad, tipo, colonia, municipio, estado):
    if n_clicks == 0:
        return ''
    input_data = pd.DataFrame({
        'Estacionamientos': [estacionamientos],
        'Superficie': [superficie],
        'Recamaras': [recamaras],
        'Baños': [banos],
        'Seguridad': [seguridad],
        'Balcon': [balcon],
        'Piscina': [piscina],
        'Jardin': [jardin],
        'Terraza': [terraza],
        'Colonia': [colonia],   
        'Municipio': [municipio],
        'Estado': [estado],
        'Tipo': [tipo],
    })

    precio_predicho = preprocess_and_predict(input_data, best_xgboost_model)
    return html.Span(f"El precio predicho de la vivienda es: $ {precio_predicho:,.2f} MXN", style={'fontSize': '20px', 'color': 'green'})
if __name__ == '__main__':
    app.run(debug=True, port=8052)
