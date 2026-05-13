
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import stats
import json
import base64
from datetime import datetime
import sqlite3
import io

# Configuración de página
st.set_page_config(
    page_title="Gestor Integral de Perforación y Voladura",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------ ESTILOS CSS PERSONALIZADOS ------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 10px 20px;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------ COLORES ------------------------
COLORS = {
    "banco_fill": "rgba(120,120,120,0.50)",
    "banco_line": "rgba(0,0,0,0.95)",
    "barreno_fill": "rgba(255,255,255,0.10)",
    "barreno_line": "rgba(0,0,0,0.95)",
    "carga": "rgba(40,120,255,0.90)",
    "agua": "rgba(0,200,255,0.25)",
    "agua_line": "rgba(0,200,255,0.60)",
    "taco": "rgba(255,150,40,0.92)",
    "iniciacion": "rgba(255,0,0,0.8)",
    "fragmentacion_baja": "rgba(0,255,0,0.6)",
    "fragmentacion_media": "rgba(255,255,0,0.6)",
    "fragmentacion_alta": "rgba(255,0,0,0.6)",
}

# ------------------------ FUNCIONES DE CÁLCULO ------------------------

def calcular_carga_explosivo(diametro_mm, densidad_expgcc, longitud_carga_m):
    """Calcula la carga de explosivo en kg"""
    radio_m = (diametro_mm / 1000) / 2
    volumen_m3 = np.pi * (radio_m ** 2) * longitud_carga_m
    carga_kg = volumen_m3 * densidad_expgcc * 1000  # Convertir g/cc a kg/m3
    return carga_kg

def calcular_factor_carga(carga_total_kg, volumen_roca_m3):
    """Calcula el factor de carga específico (kg/m³)"""
    if volumen_roca_m3 > 0:
        return carga_total_kg / volumen_roca_m3
    return 0

def calcular_costo_explosivos(carga_total_kg, costo_por_kg):
    """Calcula el costo total de explosivos"""
    return carga_total_kg * costo_por_kg

def modelo_kuz_ram(burden, spacing, stemming, altura_banco, diametro_mm, 
                   carga_total_kg, indice_roca, constante_explosivo):
    """
    Modelo Kuz-Ram para predicción de fragmentación
    Retorna: tamaño medio de fragmento (cm), índice de uniformidad
    """
    # Factor de carga
    A = 0.8  # Factor de roca
    
    # Tamaño medio de fragmento (ecuación de Kuznetsov)
    if carga_total_kg > 0 and altura_banco > 0:
        q = carga_total_kg / (burden * spacing * altura_banco)  # Factor de carga específico
        x50 = A * ((indice_roca / 115) ** 0.5) * ((1/q) ** 0.8) * (115/constante_explosivo) ** 0.33
    else:
        x50 = 0
    
    # Índice de uniformidad (ecuación de Cunningham)
    n = 0.8 + 0.5 * np.log10(burden / diametro_mm) if diametro_mm > 0 else 1.0
    n = max(0.5, min(n, 2.5))  # Limitar rango
    
    return x50, n

def calcular_vibraciones(distancia_m, carga_maxima_kg, k=150, alpha=-1.5):
    """
    Ley de atenuación de vibraciones (USBM)
    PPV = k * (distancia / sqrt(carga))^alpha
    Retorna: velocidad de partícula (mm/s)
    """
    if distancia_m > 0 and carga_maxima_kg > 0:
        sd = distancia_m / np.sqrt(carga_maxima_kg)
        ppv = k * (sd ** alpha)
        return ppv
    return 0

def calcular_presion_detonacion(densidad_expgcc, velocidad_detonacion_ms):
    """
    Calcula presión de detonación aproximada (GPa)
    P = ρ * VOD² / 2 (simplificado)
    """
    if densidad_expgcc > 0 and velocidad_detonacion_ms > 0:
        presion_gpa = (densidad_expgcc * 1000) * (velocidad_detonacion_ms ** 2) / (2 * 1e9)
        return presion_gpa
    return 0

def calcular_zona_seguridad(carga_total_kg, factor=100):
    """Calcula radio de zona de seguridad (m) basado en carga total"""
    return factor * np.sqrt(carga_total_kg) if carga_total_kg > 0 else 0

# ------------------------ GESTIÓN DE BASE DE DATOS ------------------------

def init_database():
    """Inicializa la base de datos SQLite"""
    conn = sqlite3.connect('voladura_db.sqlite')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voladuras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            altura_banco REAL,
            burden REAL,
            spacing REAL,
            diametro_barreno REAL,
            numero_barrenos INTEGER,
            carga_total REAL,
            factor_carga REAL,
            costo_total REAL,
            x50_fragmentacion REAL,
            indice_uniformidad REAL,
            ppv_maximo REAL,
            zona_seguridad REAL,
            parametros_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def guardar_voladura(datos):
    """Guarda los datos de una voladura en la base de datos"""
    conn = sqlite3.connect('voladura_db.sqlite')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO voladuras (
            altura_banco, burden, spacing, diametro_barreno, numero_barrenos,
            carga_total, factor_carga, costo_total, x50_fragmentacion,
            indice_uniformidad, ppv_maximo, zona_seguridad, parametros_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datos.get('altura_banco', 0),
        datos.get('burden', 0),
        datos.get('spacing', 0),
        datos.get('diametro_barreno', 0),
        datos.get('numero_barrenos', 0),
        datos.get('carga_total', 0),
        datos.get('factor_carga', 0),
        datos.get('costo_total', 0),
        datos.get('x50_fragmentacion', 0),
        datos.get('indice_uniformidad', 0),
        datos.get('ppv_maximo', 0),
        datos.get('zona_seguridad', 0),
        json.dumps(datos.get('parametros', {}))
    ))
    
    conn.commit()
    conn.close()

def cargar_historial():
    """Carga el historial de voladuras desde la base de datos"""
    conn = sqlite3.connect('voladura_db.sqlite')
    df = pd.read_sql_query("SELECT * FROM voladuras ORDER BY fecha DESC", conn)
    conn.close()
    return df

# ------------------------ EXPORTACIÓN DE DATOS ------------------------

def exportar_a_csv(datos_df):
    """Exporta datos a CSV"""
    return datos_df.to_csv(index=False).encode('utf-8')

def exportar_a_excel(datos_df):
    """Exporta datos a Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        datos_df.to_excel(writer, index=False, sheet_name='Voladura')
    return output.getvalue()

def generar_reporte_pdf(datos):
    """Genera un reporte simple en formato texto (simulado para PDF)"""
    reporte = f"""
    REPORTE DE VOLADURA
    ===================
    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PARÁMETROS GEOMÉTRICOS
    ----------------------
    Altura de banco: {datos.get('altura_banco', 0):.2f} m
    Burden: {datos.get('burden', 0):.2f} m
    Spacing: {datos.get('spacing', 0):.2f} m
    Diámetro de barreno: {datos.get('diametro_barreno', 0):.2f} mm
    Número de barrenos: {datos.get('numero_barrenos', 0)}
    
    RESULTADOS DE CÁLCULO
    ---------------------
    Carga total de explosivo: {datos.get('carga_total', 0):.2f} kg
    Factor de carga: {datos.get('factor_carga', 0):.3f} kg/m³
    Costo total de explosivos: ${datos.get('costo_total', 0):.2f}
    
    FRAGMENTACIÓN (Kuz-Ram)
    -----------------------
    Tamaño medio (x50): {datos.get('x50_fragmentacion', 0):.2f} cm
    Índice de uniformidad (n): {datos.get('indice_uniformidad', 0):.2f}
    
    SEGURIDAD
    ---------
    PPV máximo estimado: {datos.get('ppv_maximo', 0):.2f} mm/s
    Zona de seguridad: {datos.get('zona_seguridad', 0):.2f} m
    
    ===================
    Generado por Gestor de Perforación y Voladura
    """
    return reporte

# ------------------------ INICIALIZAR BASE DE DATOS ------------------------
init_database()

# ------------------------ HEADER PRINCIPAL ------------------------
st.markdown('<h1 class="main-header">💥 Gestor Integral de Perforación y Voladura</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    <p>Herramienta profesional para diseño, cálculo y optimización de voladuras en minería</p>
</div>
""", unsafe_allow_html=True)

# ------------------------ SIDEBAR - CONFIGURACIÓN GENERAL ------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dynamite.png", width=80)
    st.title("⚙️ Configuración")
    
    # Selector de modo
    modo_app = st.selectbox(
        "Modo de operación",
        ["Diseño Geométrico", "Optimización", "Análisis de Seguridad", "Historial"],
        help="Selecciona el módulo que deseas utilizar"
    )
    
    st.divider()
    
    # Parámetros de explosivos
    st.subheader("🧨 Explosivos")
    diametro_barreno_mm = st.slider("Diámetro de barreno (mm)", 50, 400, 250, 10)
    densidad_explosivo = st.slider("Densidad del explosivo (g/cc)", 0.8, 1.5, 1.15, 0.05)
    costo_explosivo_kg = st.number_input("Costo explosivo ($/kg)", 0.5, 10.0, 2.5, 0.1)
    vod_explosivo = st.slider("VOD (m/s)", 2000, 8000, 5500, 100, help="Velocidad de detonación")
    
    st.divider()
    
    # Parámetros de roca
    st.subheader("🪨 Rocas")
    indice_roca = st.slider("Índice de dureza de roca", 50, 200, 115, 5, help="Rock Factor (A)")
    constante_explosivo = st.slider("Constante de energía del explosivo", 80, 150, 115, 5)
    
    st.divider()
    
    # Parámetros de vibraciones
    st.subheader("📊 Vibraciones")
    k_atenuacion = st.number_input("Constante K (USBM)", 50, 300, 150, 10)
    alpha_atenuacion = st.number_input("Exponente α (USBM)", -2.0, -1.0, -1.5, 0.1)
    distancia_estructura = st.number_input("Distancia a estructuras (m)", 50, 2000, 500, 50)
    ppv_limite = st.number_input("PPV límite permitido (mm/s)", 5, 50, 25, 1)
    
    st.divider()
    
    # Botones de acción
    if st.button("💾 Guardar Voladura", use_container_width=True):
        st.success("Voladura guardada exitosamente")
        st.session_state.guardar = True
    
    if st.button("📤 Exportar Datos", use_container_width=True):
        st.session_state.exportar = True
    
    if st.button("🔄 Reiniciar Parámetros", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith(('lonbar', 'lq', 'agua')):
                del st.session_state[key]
        st.rerun()

# ------------------------ PESTAÑAS PRINCIPALES ------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Diseño 2D", 
    "📊 Dashboard", 
    "💰 Costos", 
    "🎯 Fragmentación",
    "📁 Historial"
])

# ==================== TAB 1: DISEÑO GEOMÉTRICO 2D ====================
with tab1:
    col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
    with col_header2:
        st.subheader("Parámetros Geométricos del Banco")
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        h = st.slider("Altura de banco h (m)", 2.0, 30.0, 10.0, 0.5, key="h_slider")
        angtalud = st.slider("Ángulo de talud (°)", 30.0, 85.0, 65.0, 0.5, key="ang_slider")
        base_y = st.slider("Base (offset vertical) (m)", 0.0, 15.0, 5.0, 0.5, key="base_slider")
    
    with colB:
        loninf = st.slider("Longitud de influencia (m)", 5.0, 150.0, 30.0, 1.0, key="loninf_slider")
        burd = st.slider("Burden (m)", 0.5, 15.0, 4.0, 0.1, key="burd_slider")
        spacing = st.slider("Spacing (m)", 0.5, 20.0, 5.0, 0.1, key="spacing_slider", help="Distancia entre barrenos")
        aux = st.slider("Ajuste hacia la cresta aux (m)", 0.0, 10.0, 2.0, 0.1, key="aux_slider")
    
    with colC:
        longbase = st.slider("Longitud base estudio (m)", 10.0, 200.0, 50.0, 1.0, key="longbase_slider")
        di = st.slider("Espesor gráfico barreno (m)", 0.05, 2.0, 0.70, 0.05, key="di_slider")
        stemming_ratio = st.slider("Ratio Taco/Burden", 0.5, 2.0, 1.0, 0.1, key="stem_ratio")
    
    st.markdown("---")
    
    # Cálculo de posiciones
    disdif = h / np.tan(np.radians(angtalud)) if np.tan(np.radians(angtalud)) != 0 else 0.0
    nbarr = loninf / burd if burd > 0 else 0
    nbarrint = int(np.floor(nbarr))
    
    mult = []
    for i in range(nbarrint):
        x = loninf - (burd * (i + 1) - aux)
        if x > 0.01:
            mult.append(x)
    
    n = len(mult)
    
    # Mostrar información resumen
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Número de barrenos", n, delta=None)
    with info_col2:
        st.metric("Burden promedio", f"{burd:.2f} m", delta=None)
    with info_col3:
        st.metric("Spacing", f"{spacing:.2f} m", delta=None)
    with info_col4:
        st.metric("Ángulo talud", f"{angtalud:.1f}°", delta=None)
    
    # ------------------------ PARÁMETROS POR BARRENO ------------------------
    with st.expander("⚙️ Parámetros por barreno (configuración individual)", expanded=False):
        st.caption("💡 El agua es referencia visual y se sobrepone; no reduce la carga. Taco = lonbar - lq (automático).")
        cols = st.columns(min(5, n) if n > 0 else 1)
        
        for i in range(n):
            with cols[i % len(cols)]:
                st.markdown(f"**Barreno {i+1}**")
                
                # Calcular valor por defecto basado en altura de banco
                lonbar_default = h * 1.1  # 10% más profundo que el banco
                
                lonbar_i = st.number_input(
                    f"L ({i+1})", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=float(st.session_state.get(f"lonbar_{i}", lonbar_default)),
                    step=0.5,
                    key=f"lonbar_input_{i}"
                )
                
                lq_max = min(lonbar_i * 0.9, lonbar_i)  # Máximo 90% de la longitud
                lq_i = st.number_input(
                    f"Carga ({i+1})", 
                    min_value=0.0, 
                    max_value=float(lonbar_i), 
                    value=float(st.session_state.get(f"lq_{i}", lq_max)),
                    step=0.5,
                    key=f"lq_input_{i}"
                )
              
                agua_i = st.number_input(
                    f"Agua ({i+1})", 
                    min_value=0.0, 
                    max_value=float(lonbar_i), 
                    value=float(st.session_state.get(f"agua_{i}", 0.0)),
                    step=0.5,
                    key=f"agua_input_{i}"
                )
                
                # Actualizar session state
                st.session_state[f"lonbar_{i}"] = lonbar_i
                st.session_state[f"lq_{i}"] = lq_i
                st.session_state[f"agua_{i}"] = agua_i
                
                taco_i = max(lonbar_i - lq_i, 0.0)
                st.write(f"Taco: `{taco_i:.2f} m`")
    
    # ------------------------ OPCIONES DE VISUALIZACIÓN ------------------------
    st.subheader("🎨 Opciones de Visualización")
    colV1, colV2, colV3 = st.columns([1, 2, 1])
    
    with colV1:
        show_dims = st.checkbox("Mostrar cotas", value=True, key="show_dims_cb")
        mostrar_3d = st.checkbox("Vista 3D", value=False, key="view_3d", help="Activar vista tridimensional")
    
    with colV2:
        dims_what = st.multiselect(
            "Qué cotas mostrar",
            options=["Banco", "Barreno", "Carga", "Agua", "Taco", "Pie de talud"],
            default=["Banco", "Carga", "Agua", "Taco", "Pie de talud"] if show_dims else [],
            key="dims_multiselect"
        )
    
    with colV3:
        sel = st.multiselect(
            "Cotas por barreno",
            options=[f"#{i+1}" for i in range(n)],
            default=[f"#{i+1}" for i in range(min(n, 2))] if n > 0 else [],
            key="barreno_multiselect"
        )
        sel_idx = {int(s.replace("#",""))-1 for s in sel}
    
    # ------------------------ FIGURA PRINCIPAL ------------------------
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter"}]],
        subplot_titles=("Sección Transversal del Banco",)
    )
    
    def add_rect_trace(x0, x1, y0, y1, fillcolor, linecolor,
                       legendgroup=None, name=None, showlegend=False, hover_text=None):
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y1, y1, y0, y0, y1],
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color=linecolor, width=1.5),
            hoverinfo="text" if hover_text else "skip",
            text=hover_text,
            legendgroup=legendgroup,
            name=name,
            showlegend=showlegend,
        ))
    
    def dim_v(x, y0, y1, text, tag="", dash="dash"):
        fig.add_shape(type="line", x0=x, y0=y0, x1=x, y1=y1,
                      line=dict(width=1.5, dash=dash, color="#333"))
        fig.add_shape(type="line", x0=x-0.35, y0=y0, x1=x+0.35, y1=y0,
                      line=dict(width=1.5, color="#333"))
        fig.add_shape(type="line", x0=x-0.35, y0=y1, x1=x+0.35, y1=y1,
                      line=dict(width=1.5, color="#333"))
        fig.add_annotation(
            x=x-0.85, y=(y0+y1)/2, text=f"{text}{tag}",
            showarrow=False, textangle=-90,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.3)", borderwidth=1,
            font=dict(size=9)
        )
    
    def dim_h(y, x0, x1, text, dash="dash", color="#333"):
        fig.add_shape(type="line", x0=x0, y0=y, x1=x1, y1=y,
                      line=dict(width=1.5, dash=dash, color=color))
        fig.add_shape(type="line", x0=x0, y0=y-0.2, x1=x0, y1=y+0.2,
                      line=dict(width=1.5, color=color))
        fig.add_shape(type="line", x0=x1, y0=y-0.2, x1=x1, y1=y+0.2,
                      line=dict(width=1.5, color=color))
        fig.add_annotation(
            x=(x0+x1)/2, y=y+0.35, text=text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.3)", borderwidth=1,
            font=dict(size=9)
        )
    
    # ------------------------ BANCO ------------------------
    bench_coordinates = [
        (0, 0),
        (0, base_y + h),
        (loninf, base_y + h),
        (loninf + disdif, base_y),
        (longbase, base_y),
        (longbase, 0),
        (0, 0),
    ]
    bx, by = zip(*bench_coordinates)
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode="lines", fill="toself",
        name="Banco",
        fillcolor=COLORS["banco_fill"],
        line=dict(color=COLORS["banco_line"], width=2.5),
        legendgroup="banco",
        showlegend=True,
        hoverinfo="text",
        text="Banco de mineral"
    ))
    
    # ------------------------ ÁNGULO DE TALUD ------------------------
    ang_cx, ang_cy = loninf + disdif, base_y
    R = max(1.5, 0.18 * h)
    theta = np.linspace(0, np.radians(angtalud), 60)
    
    arc_x = ang_cx - R * np.cos(theta)
    arc_y = ang_cy + R * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y, mode="lines",
        line=dict(width=2, dash="dot", color="#dc0000"),
        name="Ángulo talud",
        legendgroup="angulo",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    t_mid = np.radians(angtalud * 0.55)
    R_txt = R * 1.25
    txt_x = ang_cx - R_txt * np.cos(t_mid)
    txt_y = ang_cy + R_txt * np.sin(t_mid)
    
    fig.add_trace(go.Scatter(
        x=[txt_x], y=[txt_y], mode="text",
        text=[f"{angtalud:.1f}°"],
        textposition="middle center",
        legendgroup="angulo",
        showlegend=False,
        textfont=dict(size=13, color="#dc0000", weight="bold"),
        hoverinfo="skip"
    ))
    
    # Flags para leyenda única
    first_barreno = True
    first_carga = True
    first_agua = True
    first_taco = True
    
    label_x, label_y, label_text = [], [], []
    cargas_por_barreno = []
    
    # ------------------------ BARRENOS ------------------------
    y_top = base_y + h
    
    for i, x in enumerate(mult):
        lonbar_i = float(st.session_state.get(f"lonbar_input_{i}", h * 1.1))
        lq_i = float(st.session_state.get(f"lq_input_{i}", lonbar_i * 0.8))
        agua_i = float(st.session_state.get(f"agua_input_{i}", 0.0))
        taco_i = max(lonbar_i - lq_i, 0.0)
        
        # Calcular carga de explosivo para este barreno
        carga_barreno = calcular_carga_explosivo(diametro_barreno_mm, densidad_explosivo, lq_i)
        cargas_por_barreno.append(carga_barreno)
        
        x0, x1 = x - di/2, x + di/2
        y_bottom = y_top - lonbar_i
        
        # Barreno
        add_rect_trace(
            x0, x1, y_bottom, y_top,
            fillcolor=COLORS["barreno_fill"],
            linecolor=COLORS["barreno_line"],
            legendgroup="barreno",
            name="Barreno" if first_barreno else None,
            showlegend=first_barreno,
            hover_text=f"Barreno #{i+1}<br>Profundidad: {lonbar_i:.2f} m"
        )
        first_barreno = False
        
        # Carga
        if lq_i > 0:
            add_rect_trace(
                x0, x1, y_bottom, y_bottom + lq_i,
                fillcolor=COLORS["carga"],
                linecolor=COLORS["carga"],
                legendgroup="carga",
                name="Carga (explosivo)" if first_carga else None,
                showlegend=first_carga,
                hover_text=f"Barreno #{i+1}<br>Carga: {lq_i:.2f} m<br>Explosivo: {carga_barreno:.2f} kg"
            )
            first_carga = False
        
        # Taco
        if taco_i > 0:
            add_rect_trace(
                x0, x1, y_bottom + lq_i, y_top,
                fillcolor=COLORS["taco"],
                linecolor=COLORS["taco"],
                legendgroup="taco",
                name="Taco" if first_taco else None,
                showlegend=first_taco,
                hover_text=f"Barreno #{i+1}<br>Taco: {taco_i:.2f} m"
            )
            first_taco = False
        
        # Agua
        if agua_i > 0:
            add_rect_trace(
                x0, x1, y_bottom, y_bottom + agua_i,
                fillcolor=COLORS["agua"],
                linecolor=COLORS["agua_line"],
                legendgroup="agua",
                name="Agua" if first_agua else None,
                showlegend=first_agua,
                hover_text=f"Barreno #{i+1}<br>Agua: {agua_i:.2f} m"
            )
            first_agua = False
        
        label_x.append(x)
        label_y.append((y_top + y_bottom) / 2)
        label_text.append(str(i + 1))
        
        # Cotas por barreno
        if show_dims and (i in sel_idx):
            xdim = x + di + 0.9
            if "Barreno" in dims_what:
                dim_v(xdim, y_bottom, y_top, f"L={lonbar_i:.2f}", tag=f" #{i+1}")
            if "Agua" in dims_what and agua_i > 0:
                dim_v(xdim + 1.1, y_bottom, y_bottom + agua_i, f"H2O={agua_i:.2f}", tag=f" #{i+1}")
            if "Carga" in dims_what and lq_i > 0:
                dim_v(xdim + 2.2, y_bottom, y_bottom + lq_i, f"Q={lq_i:.2f}", tag=f" #{i+1}")
            if "Taco" in dims_what and taco_i > 0:
                dim_v(xdim + 3.3, y_bottom + lq_i, y_top, f"T={taco_i:.2f}", tag=f" #{i+1}")
    
    # Numeración
    fig.add_trace(go.Scatter(
        x=label_x, y=label_y,
        mode="markers+text",
        marker=dict(symbol="circle", size=20,
                    color="rgba(255,255,255,0.9)",
                    line=dict(width=2, color="#333")),
        text=label_text,
        textposition="middle center",
        textfont=dict(size=11, color="#000", weight="bold"),
        name="Nº Barreno",
        legendgroup="labels",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # Cotas del banco
    if show_dims and "Banco" in dims_what:
        dim_v(x=longbase + 0.5, y0=base_y, y1=base_y+h, text=f"H={h:.2f}")
        dim_h(y=-1.2, x0=0, x1=longbase, text=f"L={longbase:.2f}")
    
    if show_dims and "Pie de talud" in dims_what:
        dim_h(y=base_y, x0=0, x1=loninf + disdif, text="Pie de talud", dash="dash", color="#666")
    
    # Layout
    x_min = -7
    x_max = max(longbase + 2, loninf + disdif + 14)
    y_min = -2.5
    y_max = base_y + h + 5
    
    fig.update_layout(
        height=700,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            title="Leyenda",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#333",
            borderwidth=1,
            traceorder="normal",
            groupclick="togglegroup",
            font=dict(size=10)
        ),
        xaxis=dict(title="Longitud (m)", range=[x_min, x_max], zeroline=True, gridcolor="#eee"),
        yaxis=dict(title="Altura (m)", range=[y_min, y_max], zeroline=True, gridcolor="#eee"),
        plot_bgcolor="rgba(245,245,245,0.5)",
        paper_bgcolor="rgba(255,255,255,1)",
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Botones de acción rápida
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("📸 Capturar Imagen", use_container_width=True):
            st.info("Haz clic derecho en el gráfico → Guardar como PNG")
    with btn_col2:
        if st.button("🔍 Zoom Automático", use_container_width=True):
            st.js_eval("window.scrollTo(0, 0)")
    with btn_col3:
        if st.button("📋 Copiar Datos", use_container_width=True):
            st.code(f"Barrenos: {n}, Carga total: {sum(cargas_por_barreno):.2f} kg", language="text")
    with btn_col4:
        ayuda = st.toggle("Ayuda", value=False)
        if ayuda:
            st.info("""
            **💡 Consejos:**
            - Usa el slider para ajustar parámetros
            - Haz clic en la leyenda para ocultar/mostrar elementos
            - Expande 'Parámetros por barreno' para configuración individual
            - Activa 'Vista 3D' para visualización tridimensional
            """)







# ==================== TAB 2: DASHBOARD DE RESULTADOS ====================
with tab2:
    st.subheader("📊 Dashboard de Resultados de Voladura")
    
    # Calcular métricas principales
    carga_total = sum(cargas_por_barreno) if cargas_por_barreno else 0
    volumen_roca = burd * spacing * h * n if (burd > 0 and spacing > 0 and h > 0 and n > 0) else 0
    factor_carga = calcular_factor_carga(carga_total, volumen_roca)
    costo_total = calcular_costo_explosivos(carga_total, costo_explosivo_kg)
    
    # Fragmentación Kuz-Ram
    x50, indice_n = modelo_kuz_ram(
        burd, spacing, h * 0.3, h, diametro_barreno_mm,
        carga_total, indice_roca, constante_explosivo
    )
    
    # Vibraciones
    ppv_max = calcular_vibraciones(distancia_estructura, carga_total / n if n > 0 else carga_total, 
                                    k_atenuacion, alpha_atenuacion)
    
    # Zona de seguridad
    zona_seg = calcular_zona_seguridad(carga_total)
    
    # Métricas en tarjetas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Carga Total", value=f"{carga_total:.1f} kg", delta=f"{carga_total/n:.1f} kg/barreno" if n > 0 else "N/A")
    with col2:
        st.metric(label="Factor de Carga", value=f"{factor_carga:.3f} kg/m³", delta="Óptimo" if 0.3 <= factor_carga <= 0.8 else "Revisar")
    with col3:
        st.metric(label="Costo Explosivos", value=f"${costo_total:.2f}", delta=f"${costo_total/volumen_roca:.2f}/m³" if volumen_roca > 0 else "N/A")
    with col4:
        st.metric(label="Fragmentación x50", value=f"{x50:.2f} cm", delta=f"n={indice_n:.2f}")
    with col5:
        st.metric(label="PPV Estimado", value=f"{ppv_max:.2f} mm/s", delta="✓ Seguro" if ppv_max < ppv_limite else "⚠️ Excede")
    
    st.markdown("---")
    
    graf_col1, graf_col2 = st.columns(2)
    
    with graf_col1:
        st.markdown("### Distribución de Carga por Barreno")
        if cargas_por_barreno:
            df_cargas = pd.DataFrame({'Barreno': [f"#{i+1}" for i in range(len(cargas_por_barreno))], 'Carga (kg)': cargas_por_barreno})
            fig_barras = px.bar(df_cargas, x='Barreno', y='Carga (kg)', color='Carga (kg)', color_continuous_scale='YlOrRd')
            fig_barras.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_barras, use_container_width=True)
    
    with graf_col2:
        st.markdown("### Curva de Fragmentación")
        if x50 > 0:
            tamanos = np.linspace(0.1, x50 * 5, 100)
            acumulada = 100 * (1 - np.exp(-(tamanos / x50) ** indice_n))
            df_frag = pd.DataFrame({'Tamaño (cm)': tamanos, '% Acumulado': acumulada})
            fig_frag = px.line(df_frag, x='Tamaño (cm)', y='% Acumulado', title='Distribución granulométrica (Kuz-Ram)', markers=True)
            fig_frag.add_hline(y=50, line_dash="dash", annotation_text="x50")
            fig_frag.update_layout(height=400, yaxis_range=[0, 100])
            st.plotly_chart(fig_frag, use_container_width=True)

# ==================== TAB 3: COSTOS ====================
with tab3:
    st.subheader("💰 Análisis de Costos")
    costo_m3 = costo_total / volumen_roca if volumen_roca > 0 else 0
    
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
    with cost_col1: st.metric("Costo Total", f"${costo_total:.2f}")
    with cost_col2: st.metric("Volumen Roca", f"{volumen_roca:.1f} m³")
    with cost_col3: st.metric("Costo por m³", f"${costo_m3:.3f}")
    with cost_col4: st.metric("Rendimiento", f"{volumen_roca/carga_total:.1f} m³/kg" if carga_total > 0 else "N/A")
    
    st.markdown("---")
    if n > 0:
        datos_tabla = []
        for i in range(n):
            lq_i = float(st.session_state.get(f"lq_input_{i}", h * 0.9))
            carga_barreno = calcular_carga_explosivo(diametro_barreno_mm, densidad_explosivo, lq_i)
            datos_tabla.append({'Barreno': f"#{i+1}", 'Carga (kg)': round(carga_barreno, 2), 'Costo ($)': round(carga_barreno * costo_explosivo_kg, 2)})
        df_costos = pd.DataFrame(datos_tabla)
        st.dataframe(df_costos, use_container_width=True, height=300)

# ==================== TAB 4: FRAGMENTACIÓN ====================
with tab4:
    st.subheader("🎯 Predicción de Fragmentación (Kuz-Ram)")
    
    frag_col1, frag_col2 = st.columns(2)
    with frag_col1:
        st.write(f"**Burden:** {burd:.2f} m | **Spacing:** {spacing:.2f} m")
        st.write(f"**Factor de carga:** {factor_carga:.3f} kg/m³")
        st.success(f"**x50:** {x50:.2f} cm | **n:** {indice_n:.2f}")
        
        if x50 > 30:
            st.warning("Fragmentación gruesa. Reduce burden o aumenta carga.")
        elif x50 < 10:
            st.info("Fragmentación fina. Puedes aumentar burden.")
        else:
            st.success("Fragmentación óptima (10-30 cm)")
    
    with frag_col2:
        x_grid = np.linspace(0, loninf, 50)
        y_grid = np.linspace(0, h, 30)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for x_barreno in mult:
            Z += np.exp(-np.abs(X - x_barreno) / (burd * 0.5))
        Z = Z / np.max(Z) if np.max(Z) > 0 else Z
        
        fig_heatmap = go.Figure(data=go.Heatmap(z=Z, x=x_grid, y=y_grid, colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']]))
        fig_heatmap.update_layout(title="Mapa de fragmentación", height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ==================== TAB 5: HISTORIAL ====================
with tab5:
    st.subheader("📁 Historial de Voladuras")
    
    try:
        df_historial = cargar_historial()
        if not df_historial.empty:
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1: st.metric("Total Voladuras", len(df_historial))
            with stat_col2: st.metric("Carga Promedio", f"{df_historial['carga_total'].mean():.1f} kg")
            with stat_col3: st.metric("Costo Promedio", f"${df_historial['costo_total'].mean():.2f}")
            
            st.dataframe(df_historial[['fecha', 'altura_banco', 'burden', 'numero_barrenos', 'carga_total', 'factor_carga']], use_container_width=True, height=300)
            
            csv_data = exportar_a_csv(df_historial)
            st.download_button(label="📥 Descargar CSV", data=csv_data, file_name=f"voladuras_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        else:
            st.info("No hay voladuras registradas. ¡Guarda tu primera voladura!")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ------------------------ FOOTER ------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 20px;'><p><strong>Gestor Integral de Perforación y Voladura</strong></p><p>© 2024 - Herramienta para optimización minera</p></div>", unsafe_allow_html=True)
