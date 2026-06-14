"""
Demo interactiva con Streamlit para el sistema de detección de mensajes fraudulentos.
Interfaz completamente en español.
"""

import base64
import json
import os
import sys
from pathlib import Path

# Cargar variables de entorno desde .env (raíz del proyecto)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.examples import EXAMPLES
from src.utils import setup_logging

setup_logging()


# ---------------------------------------------------------------------------
# OCR con Mistral Pixtral
# ---------------------------------------------------------------------------

def _extract_chat_from_images(uploaded_files: list, api_key: str) -> str:
    """Envía las imágenes a Mistral Pixtral y devuelve el chat extraído como texto."""
    from mistralai.client.sdk import Mistral

    client = Mistral(api_key=api_key)

    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}

    content: list = []
    for f in uploaded_files:
        f.seek(0)
        img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode()
        suffix = Path(f.name).suffix.lower().lstrip(".")
        mime = mime_map.get(suffix, "image/jpeg")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    content.append({
        "type": "text",
        "text": (
            "Estas imágenes son capturas de pantalla de un chat (WhatsApp, SMS u otro mensajero). "
            "Extrae todos los mensajes en orden cronológico (de arriba a abajo, imagen por imagen).\n\n"
            "Reglas estrictas:\n"
            "- Una línea por mensaje\n"
            "- Formato: «Remitente: texto del mensaje» (usa «Desconocido» si no identificas quién envía)\n"
            "- Incluye emojis exactamente como aparecen en la pantalla\n"
            "- Omite fechas, horas, «visto», «entregado» y cualquier metadato del chat\n"
            "- Si hay una imagen o sticker, escribe [imagen] o [sticker] en su lugar\n"
            "- No añadas explicaciones ni texto que no esté en las capturas\n"
            "- No repitas mensajes que ya aparecieron en una imagen anterior"
        ),
    })

    response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Detector de Mensajes Fraudulentos",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto",
)


# ---------------------------------------------------------------------------
# Sidebar — Configuración de parámetros del sistema
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(
        "Ajusta los parámetros del sistema. Los valores por defecto "
        "corresponden a los umbrales calibrados durante el entrenamiento."
    )

    _sys_cfg: dict = {}

    with st.expander("🔗 Cascada de detección", expanded=False):
        st.caption("Umbrales de las 9 capas (afectan análisis de mensaje individual)")
        _sys_cfg["ml_conf_certain"] = st.slider(
            "Certeza ML (gate 1)", 0.50, 1.00, 0.95, 0.01,
            help="Confianza mínima del modelo ML para cortocircuitar las capas posteriores.",
        )
        _sys_cfg["rule_fraud_min"] = st.slider(
            "Puntuación mínima fraude (reglas)", 0, 100, 65, 1,
            help="Puntuación de reglas heurísticas mínima para confirmar fraude en gate 1.",
        )
        _sys_cfg["rule_legit_max"] = st.slider(
            "Puntuación máxima legítimo (reglas)", 0, 50, 15, 1,
            help="Puntuación de reglas máxima para confirmar mensaje legítimo en gate 1.",
        )
        _sys_cfg["emb_fraud_confirm"] = st.slider(
            "Similitud fraude (embeddings, gate 2)", 0.50, 1.00, 0.88, 0.01,
            help="Similitud semántica mínima con el banco de fraudes para confirmar en gate 2.",
        )
        _sys_cfg["emb_legit_confirm"] = st.slider(
            "Similitud legítimo (embeddings, gate 2)", 0.50, 1.00, 0.88, 0.01,
        )
        _sys_cfg["anomaly_boost_thr"] = st.slider(
            "Umbral anomalía (Isolation Forest)", 0.00, 1.00, 0.75, 0.01,
            help="Score mínimo del detector de anomalías para elevar el riesgo.",
        )
        _sys_cfg["bayes_high_thr"] = st.slider(
            "Umbral Bayes (riesgo alto)", 0.50, 1.00, 0.85, 0.01,
            help="Probabilidad posterior de fraude en la red bayesiana para nivel alto.",
        )
        _sys_cfg["cbr_high_thr"] = st.slider(
            "Umbral CBR (riesgo alto)", 0.50, 1.00, 0.86, 0.01,
            help="Score CBR mínimo (≥6/7 vecinos fraude) para nivel alto.",
        )
        _sys_cfg["meta_high_thr"] = st.slider(
            "Umbral meta-learner (alto)", 0.30, 1.00, 0.70, 0.01,
            help="Probabilidad del meta-learner para clasificar como riesgo alto.",
        )
        _sys_cfg["meta_med_thr"] = st.slider(
            "Umbral meta-learner (medio)", 0.05, 0.69, 0.35, 0.01,
            help="Probabilidad del meta-learner para clasificar como riesgo medio.",
        )

    with st.expander("💬 Análisis conversacional", expanded=False):
        st.caption("Umbrales de la detección de patrones en conversaciones")
        _sys_cfg["candidate_threshold"] = st.slider(
            "Umbral candidato", 0.05, 0.90, 0.40, 0.01,
            help="Score mínimo para considerar una ventana como candidata a patrón fraudulento.",
        )
        _sys_cfg["llm_threshold"] = st.slider(
            "Umbral LLM conversacional", 0.20, 0.95, 0.55, 0.01,
            help="Score mínimo para invocar el LLM en el análisis de una subsecuencia.",
        )
        st.caption("Pesos del score final (deben sumar ≈ 1.0)")
        _pattern_w = st.slider(
            "Peso patrón estadístico", 0.00, 1.00, 0.55, 0.05,
            help="Peso del score de patrón estadístico en el score combinado.",
        )
        _sys_cfg["pattern_weight"] = _pattern_w
        _sys_cfg["model_weight"]   = round(1.0 - _pattern_w, 2)
        st.caption(f"→ Peso modelo ML: **{_sys_cfg['model_weight']:.2f}** (complementario)")
        _sys_cfg["overlap_ratio"] = st.slider(
            "Ratio solapamiento (fusión de ventanas)", 0.10, 1.00, 0.60, 0.05,
            help="Solapamiento mínimo para fusionar dos ventanas del mismo tipo de patrón.",
        )
        st.divider()
        st.caption("Niveles de riesgo conversacional")
        _sys_cfg["critical_threshold"] = st.slider(
            "Umbral crítico", 0.60, 1.00, 0.80, 0.01,
        )
        _sys_cfg["high_threshold"] = st.slider(
            "Umbral alto", 0.30, 0.79, 0.60, 0.01,
        )
        _sys_cfg["medium_threshold"] = st.slider(
            "Umbral medio", 0.05, 0.59, 0.35, 0.01,
        )

    if st.button("↺ Restaurar valores por defecto", use_container_width=True):
        st.rerun()

    st.session_state["sys_cfg"] = _sys_cfg

# ---------------------------------------------------------------------------
# Carga de modelos (con caché)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de detección...")
def _load_predictor():
    try:
        from src.ml.predict import FraudPredictor
        return FraudPredictor(), None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Cargando analizador de conversaciones...")
def _load_conversation_analyzer():
    try:
        from src.conversation.analyzer import ConversationAnalyzer
        return ConversationAnalyzer(enable_ml=True, enable_llm=True), None
    except Exception as exc:
        return None, str(exc)


predictor, load_error = _load_predictor()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🛡️ Detector de Mensajes Fraudulentos")
st.markdown(
    """
    Analiza mensajes de SMS, WhatsApp, correo o cualquier texto en busca de señales
    de **fraude digital**: phishing, smishing, suplantación de identidad y solicitudes sospechosas.
    """
)

if load_error:
    st.warning(
        "⚠️ **Modelo no encontrado.** El análisis por reglas sigue disponible, "
        "pero para clasificación ML primero ejecuta:\n\n"
        "```bash\npython main.py train --dataset data/processed/messages.csv\n```\n\n"
        f"Detalle técnico: `{load_error}`"
    )

st.divider()

# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tab_single, tab_conv, tab_img = st.tabs([
    "📩 Mensaje individual",
    "💬 Análisis de conversación",
    "📷 Imágenes de chat",
])

# ===========================================================================
# TAB 1 — Mensaje individual
# ===========================================================================
with tab_single:
    col_input, col_examples = st.columns([3, 2])

    with col_input:
        st.subheader("✏️ Escribe o pega un mensaje")
        message_input = st.text_area(
            label="Mensaje a analizar:",
            height=160,
            placeholder="Ej: Su cuenta será bloqueada. Verifique sus datos en el siguiente enlace...",
            label_visibility="collapsed",
        )

    with col_examples:
        st.subheader("📋 Ejemplos predefinidos")
        titles = ["— Selecciona un ejemplo —"] + [ex["title"] for ex in EXAMPLES]
        selected = st.selectbox("Ejemplo:", titles, label_visibility="collapsed")
        if selected != "— Selecciona un ejemplo —":
            chosen = next((e for e in EXAMPLES if e["title"] == selected), None)
            if chosen:
                message_input = chosen["message"]
                st.caption(f"**Contexto:** {chosen['explanation']}")

    analyze = st.button("🔍 Analizar mensaje", type="primary", use_container_width=True)

    if analyze:
        if not message_input or not message_input.strip():
            st.warning("Por favor ingresa un mensaje para analizar.")
        else:
            with st.spinner("Analizando..."):
                from src.data.preprocessing import clean_text
                from src.rules.risk import analyze_risk

                risk_info = analyze_risk(message_input)
                clean = clean_text(message_input)

                if predictor is not None:
                    result = predictor.predict(message_input)
                else:
                    rl = risk_info["risk_level"]
                    recs = {
                        "low": "No se detectan señales fuertes de fraude.",
                        "medium": "El mensaje contiene señales sospechosas. Verifique la fuente antes de responder.",
                        "high": "Posible fraude. No comparta datos personales, códigos ni realice pagos.",
                    }
                    result = {
                        "original_message": message_input,
                        "preprocessed_message": clean,
                        "predicted_class": "sin modelo (solo reglas)",
                        "risk_level": rl,
                        "confidence": None,
                        "risk_score": risk_info["risk_score"],
                        "signals": risk_info["signals"],
                        "recommendation": recs[rl],
                    }

            st.divider()
            st.subheader("📊 Resultado del análisis")

            RISK_ICONS = {"low": "🟢 BAJO", "medium": "🟡 MEDIO", "high": "🔴 ALTO"}
            risk_display = RISK_ICONS.get(result["risk_level"], result["risk_level"].upper())

            c1, c2, c3 = st.columns(3)
            c1.metric("Clasificación", result["predicted_class"].upper())
            c2.metric("Nivel de riesgo", risk_display)
            c3.metric("Puntaje de riesgo", f"{result.get('risk_score', 0)}/100")

            if result.get("confidence") is not None:
                st.progress(
                    float(result["confidence"]),
                    text=f"Confianza del modelo: {result['confidence']:.1%}"
                )

            rl = result["risk_level"]
            if rl == "low":
                st.success(f"✅ {result['recommendation']}")
            elif rl == "medium":
                st.warning(f"⚠️ {result['recommendation']}")
            else:
                st.error(f"🚨 {result['recommendation']}")

            st.subheader("🚩 Señales detectadas")
            if result["signals"]:
                for signal in result["signals"]:
                    st.markdown(f"- {signal}")
            else:
                st.info("No se detectaron señales sospechosas específicas.")

            with st.expander("🔎 Ver texto preprocesado (tokens normalizados)"):
                st.code(result["preprocessed_message"] or "(vacío tras limpieza)")

            if predictor is not None:
                with st.expander("🔬 Explicación del modelo (SHAP) — top features"):
                    try:
                        from src.ml.explain import explain_prediction
                        explanation = explain_prediction(
                            message_input,
                            predictor.model,
                            predictor.vectorizer,
                            predictor.int_to_label,
                            predictor.use_manual_features,
                            top_n=10,
                        )
                        if explanation.get("shap_available"):
                            st.markdown("**Señales que más influyeron en la predicción:**")
                            for feat in explanation["top_features"]:
                                arrow = "🔴" if feat["direction"] == "fraud" else "🟢"
                                st.markdown(
                                    f"{arrow} `{feat['feature']}` — "
                                    f"{'↑ fraude' if feat['direction'] == 'fraud' else '↓ fraude'} "
                                    f"({feat['shap_value']:+.4f})"
                                )
                        else:
                            st.info(
                                "Explicación SHAP no disponible para este modelo. "
                                + explanation.get("error", "")
                            )
                    except Exception as e:
                        st.info(f"SHAP no disponible: {e}")

# ===========================================================================
# TAB 2 — Análisis de conversación
# ===========================================================================
with tab_conv:
    st.subheader("💬 Analiza una conversación completa")
    st.markdown(
        "Ingresa la conversación para detectar **subsecuencias de comportamiento sospechoso**: "
        "escalada de urgencia, suplantación de identidad, manipulación social, estafas de premio y más."
    )

    conv_format = st.radio(
        "Formato de entrada:",
        ["Un mensaje por línea (texto plano)", "JSON estructurado (con remitente y hora)"],
        horizontal=True,
    )

    if conv_format == "Un mensaje por línea (texto plano)":
        conv_placeholder = (
            "Hola, soy del banco BBVA\n"
            "Detectamos un cargo sospechoso en su cuenta\n"
            "Necesitamos verificar su identidad urgentemente\n"
            "Por favor proporcione su NIP y código OTP ahora mismo"
        )
        conv_input = st.text_area(
            "Conversación (un mensaje por línea):",
            height=200,
            placeholder=conv_placeholder,
            label_visibility="collapsed",
        )
    else:
        conv_placeholder = json.dumps(
            [
                {"text": "Hola, soy del banco BBVA", "sender": "suspect"},
                {"text": "Detectamos un cargo sospechoso", "sender": "suspect"},
                {"text": "¿Qué cargo?", "sender": "user"},
                {"text": "Proporcione su NIP urgentemente", "sender": "suspect"},
            ],
            ensure_ascii=False,
            indent=2,
        )
        conv_input = st.text_area(
            "Conversación en formato JSON:",
            height=250,
            value=conv_placeholder,
            label_visibility="collapsed",
        )

    col_opt1, col_opt2 = st.columns(2)
    use_ml  = col_opt1.checkbox("Usar modelo ML", value=True)
    use_llm = col_opt2.checkbox("Usar LLM Mistral", value=False,
                                help="Requiere MISTRAL_API_KEY. Análisis más profundo pero más lento.")

    analyze_conv = st.button(
        "🔍 Analizar conversación", type="primary", use_container_width=True
    )

    if analyze_conv:
        if not conv_input or not conv_input.strip():
            st.warning("Por favor ingresa una conversación para analizar.")
        else:
            # Parsear mensajes
            from src.conversation.models import Message as ConvMessage

            messages: list[ConvMessage] = []
            parse_error = None
            try:
                if conv_format == "Un mensaje por línea (texto plano)":
                    for line in conv_input.strip().split("\n"):
                        if line.strip():
                            messages.append(ConvMessage(text=line.strip()))
                else:
                    raw = json.loads(conv_input)
                    for item in raw:
                        if isinstance(item, str):
                            messages.append(ConvMessage(text=item))
                        else:
                            messages.append(ConvMessage(
                                text=item.get("text", ""),
                                sender=item.get("sender", "unknown"),
                                timestamp=item.get("timestamp"),
                            ))
            except Exception as exc:
                parse_error = str(exc)

            if parse_error:
                st.error(f"Error al leer la conversación: {parse_error}")
            elif len(messages) < 2:
                st.warning("Se necesitan al menos 2 mensajes para analizar una conversación.")
            else:
                with st.spinner(f"Analizando {len(messages)} mensajes..."):
                    from src.conversation.analyzer import ConversationAnalyzer

                    analyzer = ConversationAnalyzer(
                        enable_ml=use_ml,
                        enable_llm=use_llm,
                        cfg=st.session_state.get("sys_cfg", {}),
                    )
                    report = analyzer.analyze(messages)

                st.divider()
                st.subheader("📊 Resultado del análisis")

                RISK_ICONS_CONV  = {
                    "low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"
                }
                RISK_LABELS_CONV = {
                    "low": "BAJO", "medium": "MEDIO", "high": "ALTO", "critical": "CRÍTICO"
                }
                icon_g  = RISK_ICONS_CONV.get(report.overall_risk, "⚪")
                label_g = RISK_LABELS_CONV.get(report.overall_risk, report.overall_risk.upper())

                c1, c2, c3 = st.columns(3)
                c1.metric("Mensajes analizados", report.total_messages)
                c2.metric("Riesgo general", f"{icon_g} {label_g}")
                c3.metric("Patrones detectados", len(report.pattern_matches))

                if report.overall_score > 0:
                    st.progress(
                        min(report.overall_score, 1.0),
                        text=f"Score de riesgo conversacional: {report.overall_score:.0%}"
                    )

                # Recomendación global
                if report.overall_risk in ("high", "critical"):
                    st.error(
                        "🚨 **Conversación altamente sospechosa.** "
                        "Se detectaron patrones de comportamiento fraudulento. "
                        "No proporcione datos personales, credenciales ni realice pagos."
                    )
                elif report.overall_risk == "medium":
                    st.warning(
                        "⚠️ **Conversación con señales sospechosas.** "
                        "Verifique la identidad del remitente por un canal oficial antes de responder."
                    )
                else:
                    st.success("✅ No se detectaron patrones conductuales sospechosos significativos.")

                # Resumen LLM
                if report.llm_summary:
                    st.info(f"**Análisis LLM:** {report.llm_summary}")

                # Patrones detectados
                if report.pattern_matches:
                    st.subheader("🚩 Patrones conductuales detectados")
                    for match in report.pattern_matches:
                        risk_icon  = RISK_ICONS_CONV.get(match.risk_level, "⚪")
                        risk_label = RISK_LABELS_CONV.get(match.risk_level, match.risk_level.upper())
                        with st.expander(
                            f"{risk_icon} **{match.pattern_type}** {match.span}  "
                            f"— {risk_label} ({match.confidence:.0%})"
                        ):
                            st.markdown(f"**Descripción:** {match.pattern_description}")
                            st.markdown(
                                f"**Score de reglas:** {match.rule_score:.0%}  |  "
                                f"**Confianza final:** {match.confidence:.0%}"
                            )
                            if match.llm_analysis:
                                if match.llm_analysis.get("explanation"):
                                    st.markdown(f"**Análisis LLM:** {match.llm_analysis['explanation']}")
                                if match.llm_analysis.get("tactics"):
                                    tacs = ", ".join(match.llm_analysis["tactics"])
                                    st.markdown(f"**Tácticas:** {tacs}")
                            st.markdown("**Mensajes de la subsecuencia:**")
                            for i, msg in enumerate(match.messages):
                                rs = msg.individual_risk.get("risk_score", 0) if msg.individual_risk else 0
                                risk_badge = f" ⚠ {rs}/100" if rs >= 25 else ""
                                st.markdown(
                                    f"- `[{match.start_idx + i}]` **({msg.sender})** "
                                    f"{msg.text}{risk_badge}"
                                )

                # Gráfico de riesgo por mensaje
                if report.messages:
                    st.subheader("📈 Riesgo por mensaje")
                    chart_data = {
                        f"[{i}] {m.text[:30]}…" if len(m.text) > 30 else f"[{i}] {m.text}": (
                            m.individual_risk.get("risk_score", 0) if m.individual_risk else 0
                        )
                        for i, m in enumerate(report.messages)
                    }
                    import pandas as pd
                    df_chart = pd.DataFrame(
                        list(chart_data.items()),
                        columns=["Mensaje", "Risk Score"]
                    ).set_index("Mensaje")
                    st.bar_chart(df_chart)

# ===========================================================================
# TAB 3 — Imágenes de chat
# ===========================================================================
with tab_img:
    st.subheader("📷 Extrae un chat desde capturas de pantalla")
    st.markdown(
        "Sube una o varias capturas del chat (en orden cronológico). "
        "La IA leerá el texto de cada imagen y te lo mostrará en un editor "
        "para que puedas corregir errores antes de analizarlo."
    )

    api_key_img = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key_img:
        st.warning(
            "⚠️ **Se requiere MISTRAL_API_KEY** para leer las imágenes. "
            "Configúrala en el entorno antes de iniciar Streamlit:\n\n"
            "```bash\nexport MISTRAL_API_KEY=tu_clave\nstreamlit run app/streamlit_app.py\n```"
        )

    uploaded = st.file_uploader(
        "Capturas de pantalla del chat (PNG, JPG, WEBP):",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} imagen(es) cargada(s) — previsualización:**")
        preview_cols = st.columns(min(len(uploaded), 3))
        for idx, uf in enumerate(uploaded):
            with preview_cols[idx % 3]:
                uf.seek(0)
                st.image(uf, caption=uf.name, use_container_width=True)

        st.divider()

        col_extract, col_clear = st.columns([3, 1])
        do_extract = col_extract.button(
            "🤖 Extraer texto con IA",
            disabled=not api_key_img,
            use_container_width=True,
        )
        if col_clear.button("🗑 Limpiar", use_container_width=True):
            st.session_state.pop("img_ocr_text", None)

        if do_extract:
            with st.spinner("Analizando imágenes con Mistral Pixtral…"):
                try:
                    ocr_result = _extract_chat_from_images(uploaded, api_key_img)
                    st.session_state["img_ocr_text"] = ocr_result
                    st.success("✓ Texto extraído correctamente.")
                except Exception as exc:
                    st.error(f"Error al procesar las imágenes: {exc}")

    # --- Editor de texto extraído ---
    if "img_ocr_text" in st.session_state or (uploaded and not api_key_img):
        st.markdown("### ✏️ Texto extraído — edítalo si hay errores")
        st.caption(
            "Cada línea es un mensaje. Puedes corregir palabras mal reconocidas, "
            "eliminar líneas que no son mensajes o reorganizar el orden."
        )

        default_text = st.session_state.get(
            "img_ocr_text",
            "Remitente: escribe aquí el texto si lo copiaste manualmente…",
        )
        edited_ocr = st.text_area(
            "Chat extraído:",
            value=default_text,
            height=280,
            key="img_text_editor",
            label_visibility="collapsed",
        )

        st.divider()

        col_o1, col_o2 = st.columns(2)
        use_ml_img  = col_o1.checkbox("Usar modelo ML", value=True, key="img_use_ml")
        use_llm_img = col_o2.checkbox(
            "Usar LLM Mistral", value=bool(api_key_img), key="img_use_llm",
            help="Requiere MISTRAL_API_KEY. Análisis semántico más profundo."
        )

        analyze_img = st.button(
            "🔍 Analizar esta conversación",
            type="primary",
            use_container_width=True,
            key="img_analyze_btn",
        )

        if analyze_img:
            lines = [l.strip() for l in edited_ocr.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                st.warning("Se necesitan al menos 2 mensajes para analizar una conversación.")
            else:
                from src.conversation.models import Message as ConvMessage
                from src.conversation.analyzer import ConversationAnalyzer

                # Parsear "Remitente: texto" o línea simple
                img_messages: list[ConvMessage] = []
                for line in lines:
                    if ": " in line:
                        sender, _, text = line.partition(": ")
                        img_messages.append(ConvMessage(text=text.strip(), sender=sender.strip()))
                    else:
                        img_messages.append(ConvMessage(text=line))

                with st.spinner(f"Analizando {len(img_messages)} mensajes…"):
                    img_analyzer = ConversationAnalyzer(
                        enable_ml=use_ml_img,
                        enable_llm=use_llm_img,
                        cfg=st.session_state.get("sys_cfg", {}),
                    )
                    img_report = img_analyzer.analyze(img_messages)

                st.divider()
                st.subheader("📊 Resultado del análisis")

                _ICONS  = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}
                _LABELS = {"low": "BAJO", "medium": "MEDIO", "high": "ALTO", "critical": "CRÍTICO"}
                r_icon  = _ICONS.get(img_report.overall_risk, "⚪")
                r_label = _LABELS.get(img_report.overall_risk, img_report.overall_risk.upper())

                m1, m2, m3 = st.columns(3)
                m1.metric("Mensajes analizados", img_report.total_messages)
                m2.metric("Riesgo general", f"{r_icon} {r_label}")
                m3.metric("Patrones detectados", len(img_report.pattern_matches))

                if img_report.overall_score > 0:
                    st.progress(
                        min(img_report.overall_score, 1.0),
                        text=f"Score de riesgo: {img_report.overall_score:.0%}",
                    )

                if img_report.overall_risk in ("high", "critical"):
                    st.error(
                        "🚨 **Conversación altamente sospechosa.** "
                        "No proporcione datos personales ni realice pagos."
                    )
                elif img_report.overall_risk == "medium":
                    st.warning(
                        "⚠️ **Señales sospechosas detectadas.** "
                        "Verifique la identidad del remitente por un canal oficial."
                    )
                else:
                    st.success("✅ No se detectaron patrones conductuales sospechosos significativos.")

                if img_report.llm_summary:
                    st.info(f"**Análisis LLM:** {img_report.llm_summary}")

                if img_report.pattern_matches:
                    st.subheader("🚩 Patrones conductuales detectados")
                    for match in img_report.pattern_matches:
                        r_i = _ICONS.get(match.risk_level, "⚪")
                        r_l = _LABELS.get(match.risk_level, match.risk_level.upper())
                        with st.expander(
                            f"{r_i} **{match.pattern_type}** {match.span} — {r_l} ({match.confidence:.0%})"
                        ):
                            st.markdown(f"**Descripción:** {match.pattern_description}")
                            if match.llm_analysis and match.llm_analysis.get("explanation"):
                                st.markdown(f"**Análisis LLM:** {match.llm_analysis['explanation']}")
                            st.markdown("**Mensajes involucrados:**")
                            for i, msg in enumerate(match.messages):
                                rs = msg.individual_risk.get("risk_score", 0) if msg.individual_risk else 0
                                badge = f" ⚠ {rs}/100" if rs >= 25 else ""
                                st.markdown(
                                    f"- `[{match.start_idx + i}]` **({msg.sender})** {msg.text}{badge}"
                                )

                if img_report.messages:
                    st.subheader("📈 Riesgo por mensaje")
                    import pandas as pd
                    chart_img = {
                        f"[{i}] {m.text[:28]}…" if len(m.text) > 28 else f"[{i}] {m.text}": (
                            m.individual_risk.get("risk_score", 0) if m.individual_risk else 0
                        )
                        for i, m in enumerate(img_report.messages)
                    }
                    df_img = pd.DataFrame(
                        list(chart_img.items()), columns=["Mensaje", "Risk Score"]
                    ).set_index("Mensaje")
                    st.bar_chart(df_img)


# ---------------------------------------------------------------------------
# Pie de página
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ **Aviso:** Este sistema es una herramienta de apoyo académico. "
    "No reemplaza el juicio humano ni debe usarse para monitorear "
    "comunicaciones privadas sin consentimiento explícito. "
    "Cualquier dato real debe anonimizarse antes de ser procesado."
)
