"""
Mensajes de ejemplo para la demo de Streamlit.
Cubre los principales patrones de fraude digital en español.
"""

EXAMPLES: list[dict] = [
    {
        "title": "Bloqueo de cuenta bancaria",
        "message": (
            "ALERTA: Su cuenta bancaria será bloqueada en 24 horas por actividad inusual. "
            "Verifique sus datos de inmediato en: http://banco-seguro-verificacion.com/acceder"
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Phishing bancario. Combina urgencia, amenaza de bloqueo y enlace falso "
            "para robar credenciales. Patrón muy común en México y LATAM."
        ),
    },
    {
        "title": "Premio falso",
        "message": (
            "¡Felicidades! Usted ha sido seleccionado como ganador de $50,000 pesos en nuestro sorteo. "
            "Para reclamar su premio envíe su número de tarjeta, CVV y fecha de vencimiento al 5555-1234."
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Smishing clásico de premio inexistente. Solicita datos sensibles de tarjeta "
            "bajo pretexto de cobrar un premio."
        ),
    },
    {
        "title": "Solicitud de código OTP",
        "message": (
            "Mándame el código que te llegó por SMS, es urgente, "
            "necesito confirmar tu cuenta ahora mismo o la pierdes."
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Suplantación de identidad con ingeniería social. Solicita OTP de autenticación "
            "con urgencia extrema para tomar control de la cuenta víctima."
        ),
    },
    {
        "title": "Mensaje legítimo — Pedido listo",
        "message": "Hola, ya está listo el pedido que solicitaste. Pasa a recogerlo cuando gustes.",
        "expected_category": "legitimate",
        "explanation": (
            "Notificación de pedido sin señales de fraude. "
            "Sin urgencia, sin solicitudes de datos, sin enlaces sospechosos."
        ),
    },
    {
        "title": "Transferencia y datos personales",
        "message": (
            "Transferimos el dinero a su cuenta. Para confirmar la transacción ingrese "
            "su CLABE interbancaria, NIP y contraseña en: https://bit.ly/confirmar-tx-banco"
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Phishing financiero. Simula una transferencia recibida para justificar "
            "la solicitud de credenciales bancarias completas."
        ),
    },
    {
        "title": "Oferta de empleo falsa",
        "message": (
            "¡Trabaja desde casa y gana $3,000 diarios! Sin experiencia necesaria. "
            "Solo necesitas tu número de cuenta para recibir tu primer pago anticipado. "
            "Regístrate AHORA: www.empleos-rapidos-mx.net/registro"
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Fraude de oferta de empleo. Promesa de dinero fácil, solicita cuenta bancaria "
            "y dirige a un sitio web sospechoso."
        ),
    },
    {
        "title": "Falsa notificación de paquete",
        "message": (
            "Su paquete #MX8273 no pudo ser entregado. "
            "Reprogramar entrega (cargo de $12): http://correos-mx.rastreo-entrega.com/MX8273"
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Smishing de paquetería falso. Imita a servicios de correo para cobrar un cargo "
            "pequeño y capturar datos de tarjeta."
        ),
    },
    {
        "title": "Mensaje legítimo — Recordatorio médico",
        "message": (
            "Recordatorio: tienes una cita médica mañana martes a las 10:00 AM en Consultorios del Sur. "
            "Si necesitas cancelar llama al 55 1234 5678."
        ),
        "expected_category": "legitimate",
        "explanation": (
            "Recordatorio de cita legítimo. Tiene número de teléfono pero en contexto "
            "de cancelación de cita, sin solicitud de datos ni urgencia artificial."
        ),
    },
    {
        "title": "Suplantación bancaria (SAT)",
        "message": (
            "SAT: Tiene un saldo a favor de $12,450.00 pendiente de devolución. "
            "Para proceder ingrese su RFC, CURP y datos bancarios en: sat-devolucion.tramite-gov.com"
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Fraude fiscal con suplantación del SAT. Promete devolución de impuestos "
            "para obtener identidad completa y datos bancarios. URL falsa."
        ),
    },
    {
        "title": "Venta falsa en marketplace",
        "message": (
            "Hola, vi tu anuncio. Te compro el artículo pero necesito que primero "
            "me envíes tu número de cuenta completo y una foto de tu INE para "
            "poder hacer la transferencia desde mi banco empresarial."
        ),
        "expected_category": "fraudulent",
        "explanation": (
            "Fraude en compraventa. El pretexto de 'banco empresarial' se usa para "
            "solicitar datos de identidad antes de cualquier pago real."
        ),
    },
]
