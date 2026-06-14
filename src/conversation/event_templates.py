"""
Biblioteca de templates para el simulador DES de conversaciones fraudulentas.

Organización:
  CONTEXT_POOLS   — valores posibles de cada variable de contexto por tipo de fraude
  TEMPLATES       — mensajes por (event_type, fraud_type), 3-5 variantes
  LEGIT_TEMPLATES — mensajes legítimos por categoría temática

Los placeholders en los templates usan formato Python {clave} y se resuelven
con el contexto compartido de la conversación (mismo banco, misma credencial,
etc. para todos los mensajes de una sesión de fraude).
"""

# ---------------------------------------------------------------------------
# Pools de contexto — valores concretos por tipo de fraude
# ---------------------------------------------------------------------------

CONTEXT_POOLS: dict[str, dict[str, list[str]]] = {
    "phishing_bancario": {
        "banco":         ["BBVA", "Santander", "HSBC", "Banamex", "Banorte",
                          "Citibanamex", "Scotiabank", "BanBajío"],
        "credencial":    ["NIP", "contraseña", "código OTP", "número de tarjeta",
                          "PIN", "clave dinámica", "token de seguridad"],
        "url":           ["bbva-verifica.com", "santander-seguro.mx",
                          "hsbc-mx.net/login", "banca-segura.com/acceso",
                          "verificacion-cuenta.mx", "banco-online.net/verify"],
        "phone":         ["800 123 4567", "55 1234 5678", "+52 800 000 1234",
                          "01 800 555 0000"],
        "monto":         ["$5,000 MXN", "$2,500 pesos", "$10,000",
                          "$1,800 MXN", "el saldo completo de su cuenta"],
        "accion":        ["bloqueada", "suspendida", "comprometida",
                          "en revisión de seguridad", "temporalmente deshabilitada"],
        "urgencia_frase":["inmediatamente", "en las próximas 24 horas",
                          "hoy mismo antes de las 6 pm", "en los próximos 30 minutos",
                          "antes de que cierre el sistema"],
        "servicio":      ["tarjeta de crédito", "cuenta de débito",
                          "acceso a banca en línea", "cuenta maestra"],
    },
    "sat_devolucion": {
        "entidad":       ["SAT", "el Servicio de Administración Tributaria",
                          "SHCP", "Hacienda"],
        "credencial":    ["RFC", "CURP", "contraseña del portal SAT",
                          "e.firma", "datos de su declaración fiscal"],
        "url":           ["sat-tramites.gob.mx.return.com", "devoluciones-sat.com",
                          "tramite-fiscal.mx", "sat-devolucion.com/claim"],
        "monto":         ["$12,500 MXN", "$8,000 pesos", "$4,200 de devolución",
                          "$6,750 MXN en impuestos retenidos"],
        "urgencia_frase":["antes del cierre del ejercicio fiscal",
                          "en 48 horas hábiles", "este mes calendario",
                          "antes del 31 de este mes"],
        "accion":        ["retenida en el sistema", "pendiente de validar",
                          "en proceso de liberación", "bloqueada por verificación"],
    },
    "paqueteria_falsa": {
        "paqueteria":    ["FedEx", "DHL", "Correos de México", "Estafeta",
                          "UPS", "MercadoEnvíos"],
        "credencial":    ["número de guía", "código de paquete",
                          "datos de envío", "número de orden"],
        "url":           ["fedex-mx.tracking.com", "dhl-seguimiento.com",
                          "correosdemexico.rastreo.net", "paquete-retenido.mx"],
        "phone":         ["800 999 3333", "55 8888 7777", "01 800 DHL MEXICO"],
        "monto":         ["$350 pesos de almacenaje", "$199 de arancel",
                          "$450 MXN de despacho aduanal", "$280 de impuesto"],
        "urgencia_frase":["en 24 horas o será devuelto", "antes del mediodía",
                          "hoy mismo para evitar cargos adicionales",
                          "en 2 días o se regresará al remitente"],
        "accion":        ["detenido en aduana", "retenido en almacén",
                          "en espera de pago", "bloqueado por aduana"],
    },
    "premio_oxxo": {
        "empresa":       ["OXXO", "Walmart", "Liverpool", "Soriana",
                          "Coppel", "Bodega Aurrerá", "Suburbia"],
        "credencial":    ["CURP", "INE", "número de afiliado",
                          "datos bancarios para el depósito", "cuenta CLABE"],
        "url":           ["oxxo-premio.com/reclamar", "walmart-sorteo.mx",
                          "liverpool-ganador.com", "sorteo-especial.mx/claim"],
        "phone":         ["800 555 6789", "55 4444 3333", "01 800 SORTEOS"],
        "monto":         ["$50,000 pesos", "$25,000 MXN", "un automóvil 0 km",
                          "$15,000 en mercancías", "un viaje todo incluido"],
        "urgencia_frase":["en 72 horas o pierde el premio", "antes del viernes",
                          "hoy es el último día para reclamar",
                          "el premio vence esta semana"],
        "accion":        ["acumulado en el sistema", "esperando ser reclamado",
                          "reservado a su nombre", "pendiente de entrega"],
    },
    "solicitud_otp": {
        "banco":         ["BBVA", "Santander", "Bancomer", "HSBC", "Banorte"],
        "credencial":    ["código OTP", "NIP de un solo uso", "token",
                          "código de verificación", "clave temporal"],
        "url":           ["banca-otp.com", "verificar-codigo.mx",
                          "token-seguro.com", "auth-banco.net"],
        "phone":         ["800 000 9999", "55 7777 8888"],
        "accion":        ["en proceso de verificación", "pendiente de confirmar",
                          "bloqueada por seguridad"],
        "urgencia_frase":["en los próximos 5 minutos", "antes de que expire",
                          "ahora mismo", "en 10 minutos o la operación se cancela"],
        "servicio":      ["transferencia bancaria", "compra en línea",
                          "cambio de contraseña", "actualización de datos"],
    },
    "empleo_falso": {
        "empresa":       ["Amazon México", "Walmart Corporate", "empresa internacional",
                          "compañía de outsourcing", "firma de consultoría"],
        "credencial":    ["CURP", "comprobante de domicilio", "número de cuenta CLABE",
                          "datos bancarios", "RFC", "acta de nacimiento"],
        "url":           ["amazon-empleos.com", "trabajo-remoto-mx.com",
                          "bolsa-empleo-internacional.net", "hr-vacantes.mx"],
        "monto":         ["$25,000 mensuales", "$18,000 pesos al mes",
                          "$350 USD semanales", "$500 dólares por semana"],
        "urgencia_frase":["antes del lunes", "esta semana para iniciar",
                          "los lugares son limitados", "el proceso de selección cierra hoy"],
        "accion":        ["seleccionado en el proceso", "preseleccionado",
                          "aprobado en la primera fase"],
    },
    "compraventa_fraude": {
        "plataforma":    ["Facebook Marketplace", "MercadoLibre", "OLX",
                          "Vivanuncios", "Craigslist MX"],
        "credencial":    ["datos bancarios", "número de cuenta",
                          "CLABE interbancaria", "comprobante de depósito"],
        "url":           ["mp-pago-seguro.com", "mercadolibre-garantia.mx",
                          "fb-marketplace-escrow.com", "pago-seguro-mx.net"],
        "monto":         ["$3,500 pesos", "$8,000 MXN", "$1,200 de anticipo",
                          "$500 de depósito de garantía"],
        "urgencia_frase":["hoy mismo o lo vendo a otro comprador",
                          "el artículo ya tiene otros interesados",
                          "solo por hoy a ese precio", "el envío sale mañana"],
        "accion":        ["apartado a su nombre", "en espera de pago",
                          "reservado temporalmente", "listo para enviar"],
    },
}

# ---------------------------------------------------------------------------
# Templates por (tipo de evento, tipo de fraude)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, list[str]]] = {

    # ------------------------------------------------------------------
    # CONTACT_INIT — primer contacto, aparentemente inocente
    # ------------------------------------------------------------------
    "contact_init": {
        "phishing_bancario": [
            "Estimado cliente, le contactamos del área de seguridad de {banco}.",
            "Buenos días, le habla un asesor autorizado de {banco}, ¿tiene un momento?",
            "Hola, soy el representante de atención al cliente de {banco}.",
            "Buenas tardes, le comunicamos del departamento de seguridad de {banco}.",
        ],
        "sat_devolucion": [
            "Estimado contribuyente, le contactamos de {entidad}.",
            "Buenos días, le habla un funcionario de {entidad} respecto a su declaración.",
            "Buenas tardes, le comunicamos de {entidad} sobre su situación fiscal.",
            "Le habla {entidad}, hemos detectado información pendiente en su expediente.",
        ],
        "paqueteria_falsa": [
            "Le informamos que {paqueteria} tiene un paquete a su nombre.",
            "Buen día, le contactamos de {paqueteria} respecto a un envío pendiente.",
            "Hola, somos el servicio de atención al cliente de {paqueteria}.",
            "Tiene un paquete en proceso de entrega con {paqueteria}.",
        ],
        "premio_oxxo": [
            "¡Felicidades! Ha sido seleccionado como ganador en el sorteo de {empresa}.",
            "Le informamos que resultó ganador de nuestro concurso mensual en {empresa}.",
            "¡Enhorabuena! Su número fue el ganador del gran sorteo de {empresa}.",
            "Le contactamos de {empresa} para informarle de una gran noticia.",
        ],
        "solicitud_otp": [
            "Le contactamos del área de seguridad de {banco}.",
            "Hola, le llama el sistema de verificación de {banco}.",
            "Buen día, detectamos actividad en su cuenta de {banco}.",
            "Somos el equipo de seguridad digital de {banco}.",
        ],
        "empleo_falso": [
            "Le escribimos porque su perfil fue seleccionado para una oportunidad en {empresa}.",
            "Buenos días, revisamos su CV y nos interesa contactarle de {empresa}.",
            "Hola, somos reclutadores de {empresa} y encontramos su perfil muy interesante.",
            "Le contactamos de recursos humanos de {empresa} sobre una vacante.",
        ],
        "compraventa_fraude": [
            "Hola, vi su anuncio en {plataforma} y me interesa el artículo.",
            "Buenos días, estoy interesado en lo que publicó en {plataforma}.",
            "Hola, ¿todavía tiene disponible lo que anunció en {plataforma}?",
            "Vi su producto en {plataforma} y me interesa comprarlo.",
        ],
        "_default": [
            "Hola, le contactamos por un asunto importante.",
            "Buenos días, necesitamos hablar con usted sobre algo urgente.",
            "Le comunicamos respecto a un tema que requiere su atención.",
        ],
    },

    # ------------------------------------------------------------------
    # TRUST_BUILD — genera confianza, suplanta identidad oficial
    # ------------------------------------------------------------------
    "trust_build": {
        "phishing_bancario": [
            "Para su tranquilidad, le recordamos que {banco} nunca le pedirá contraseñas por teléfono, salvo en este proceso de verificación interna.",
            "Nuestros registros muestran que usted es cliente desde hace varios años y valoramos su lealtad a {banco}.",
            "Este es un proceso estándar de seguridad que {banco} realiza para proteger a sus clientes más importantes.",
            "Le aseguramos que esta llamada está siendo grabada y monitoreada por el área de cumplimiento de {banco}.",
        ],
        "sat_devolucion": [
            "Este proceso es completamente oficial y puede verificarlo en el portal de {entidad}.",
            "Su RFC ha sido verificado en nuestros sistemas y cumple con todos los requisitos.",
            "El trámite que le informamos está respaldado por la resolución miscelánea fiscal vigente.",
            "Tenemos registro de sus declaraciones de los últimos tres años y todo está en orden.",
        ],
        "paqueteria_falsa": [
            "Tenemos su información de contacto registrada en nuestro sistema de {paqueteria}.",
            "El paquete fue enviado desde otra ciudad y ya está en la oficina de {paqueteria} más cercana a usted.",
            "Nuestro sistema de rastreo confirma que el envío llegó esta mañana.",
            "Este es un paquete certificado de {paqueteria}, por eso requerimos su validación.",
        ],
        "premio_oxxo": [
            "Su número fue registrado en nuestra base de datos de participantes verificados de {empresa}.",
            "Este sorteo fue validado por la Secretaría de Gobernación con permiso número SEGOB/2024.",
            "Tenemos respaldo legal del concurso en {empresa} para garantizar la entrega de premios.",
            "Miles de participantes entran cada mes pero solo usted fue el afortunado ganador de {empresa}.",
        ],
        "solicitud_otp": [
            "Nuestro sistema de {banco} detectó un intento de acceso desde un dispositivo no reconocido.",
            "Para proteger su cuenta, necesitamos confirmar su identidad mediante verificación en dos pasos.",
            "Este es el procedimiento estándar de {banco} ante cualquier actividad sospechosa.",
            "Su seguridad es nuestra prioridad en {banco}, por eso aplicamos este protocolo.",
        ],
        "empleo_falso": [
            "Su perfil destaca entre más de 500 candidatos que aplicaron a esta vacante en {empresa}.",
            "La posición es completamente remota y los beneficios incluyen seguro médico desde el primer día.",
            "{empresa} es una empresa líder con más de 20 años de experiencia y presencia en 15 países.",
            "Hemos revisado sus referencias y todo está en orden para avanzar en el proceso.",
        ],
        "compraventa_fraude": [
            "Soy una persona seria, puede verificar mi historial de transacciones en {plataforma}.",
            "He realizado más de 50 compras exitosas en {plataforma} sin ningún problema.",
            "Prefiero usar el sistema de pago seguro de {plataforma} para mayor protección de ambos.",
            "Puedo enviar foto de mi identificación para que se sienta más seguro con la transacción.",
        ],
        "_default": [
            "Somos una institución oficial y este proceso está completamente respaldado.",
            "Puede verificar nuestra información en el portal oficial, todo es completamente legítimo.",
            "Este proceso es estándar y miles de personas lo han completado sin ningún problema.",
        ],
    },

    # ------------------------------------------------------------------
    # PROBLEM_ANNOUNCE — anuncia el problema central del fraude
    # ------------------------------------------------------------------
    "problem_announce": {
        "phishing_bancario": [
            "Hemos detectado un acceso no autorizado a su {servicio} de {banco}.",
            "Su cuenta en {banco} ha sido {accion} por actividad inusual.",
            "Nuestro sistema de seguridad de {banco} marcó su {servicio} como {accion}.",
            "Se realizó un intento de fraude en su cuenta de {banco} y la tuvimos que {accion}.",
        ],
        "sat_devolucion": [
            "Tiene una devolución de impuestos de {monto} que lleva meses {accion} en nuestro sistema.",
            "Detectamos que su declaración tiene un saldo a favor de {monto} sin reclamar.",
            "Su reembolso de {monto} está {accion} porque hay un dato de contacto desactualizado.",
            "El SAT tiene pendiente un crédito fiscal de {monto} a su nombre {accion}.",
        ],
        "paqueteria_falsa": [
            "Su paquete está {accion} en nuestra bodega por un importe pendiente de {monto}.",
            "El envío de {paqueteria} fue detenido en aduana y requiere un pago de {monto}.",
            "Hay un problema con su paquete: está {accion} y necesita liquidar {monto} de arancel.",
            "El paquete no pudo ser entregado y ahora está {accion} con un cargo de {monto}.",
        ],
        "premio_oxxo": [
            "Su premio de {monto} está {accion} en nuestros sistemas de {empresa}.",
            "El monto de {monto} ganado en {empresa} no ha podido ser transferido porque falta verificar su identidad.",
            "Para liberar su premio de {monto} en {empresa} necesitamos confirmar algunos datos.",
            "El ganador de {monto} en {empresa} debe validar su participación antes de recibir el premio.",
        ],
        "solicitud_otp": [
            "Hay una operación de {monto} en proceso desde su cuenta que no fue autorizada por usted.",
            "Alguien intentó realizar una transferencia de {monto} usando sus credenciales.",
            "Detectamos que su {servicio} está siendo utilizado desde un dispositivo desconocido.",
            "Su cuenta muestra un cargo no reconocido de {monto} que debemos cancelar ahora.",
        ],
        "empleo_falso": [
            "La empresa requiere un depósito inicial de {monto} para el kit de trabajo remoto.",
            "Para formalizar su contratación necesitamos ciertos documentos y un depósito de garantía de {monto}.",
            "El proceso requiere que tramite una membresía corporativa de {monto} para tener acceso a las herramientas.",
            "Hay un cargo administrativo de {monto} para procesar su expediente en {empresa}.",
        ],
        "compraventa_fraude": [
            "Necesito que el pago llegue hoy, otro comprador también está interesado en el artículo.",
            "El artículo vale {monto} y necesito confirmación de pago antes de separarlo.",
            "El precio final es {monto} incluyendo el envío, pero solo acepto transferencia.",
            "Para apartar el artículo necesito {monto} de anticipo, el resto al recibir.",
        ],
        "_default": [
            "Hay un problema con su cuenta que requiere atención inmediata.",
            "Detectamos una irregularidad que debemos resolver cuanto antes.",
            "Existe un cargo pendiente que necesita verificar y pagar.",
        ],
    },

    # ------------------------------------------------------------------
    # URGENCY_INJECT — inyecta presión temporal
    # ------------------------------------------------------------------
    "urgency_inject": {
        "phishing_bancario": [
            "Debe verificar su identidad {urgencia_frase} o su cuenta seguirá bloqueada.",
            "Tiene {urgencia_frase} para resolver esto o su {servicio} será cancelado permanentemente.",
            "El sistema cerrará este ticket {urgencia_frase}, actúe ahora para proteger su cuenta en {banco}.",
            "Si no completamos la verificación {urgencia_frase}, el banco deberá emitir una nueva tarjeta con demora de 15 días.",
        ],
        "sat_devolucion": [
            "La devolución expira {urgencia_frase} si no completa el trámite.",
            "El sistema fiscal cierra los expedientes vencidos, el suyo expira {urgencia_frase}.",
            "Si no reclama su devolución de {monto} {urgencia_frase}, los fondos regresan a Hacienda.",
            "El plazo legal para recibir su crédito fiscal vence {urgencia_frase}.",
        ],
        "paqueteria_falsa": [
            "El paquete será devuelto al remitente {urgencia_frase} si no se pagan los {monto} de arancel.",
            "La bodega cobra {monto} diarios de almacenaje, debe resolver esto {urgencia_frase}.",
            "Si no paga {urgencia_frase}, el paquete será destruido según el reglamento aduanal.",
            "Tiene {urgencia_frase} para liquidar los {monto} antes de que el paquete se remate.",
        ],
        "premio_oxxo": [
            "El premio de {monto} en {empresa} debe reclamarse {urgencia_frase} o se invalida.",
            "Solo tiene {urgencia_frase} para verificar su participación y reclamar su premio.",
            "Si no confirma sus datos {urgencia_frase}, el premio pasa al siguiente ganador.",
            "La empresa {empresa} tiene política de 72 horas, su tiempo vence {urgencia_frase}.",
        ],
        "solicitud_otp": [
            "Tiene {urgencia_frase} para confirmar el código antes de que la operación se procese.",
            "El código OTP vence en 5 minutos, necesita proporcionarlo {urgencia_frase}.",
            "La transacción se completará automáticamente {urgencia_frase} si no la cancela.",
            "Para detener el cargo de {monto} debe actuar {urgencia_frase}.",
        ],
        "empleo_falso": [
            "La oferta es solo para candidatos que confirmen {urgencia_frase}, los lugares son limitados.",
            "El proceso de incorporación inicia {urgencia_frase} y necesitamos su confirmación.",
            "Hay otro candidato interesado, debe confirmar su participación {urgencia_frase}.",
            "La empresa necesita su respuesta {urgencia_frase} para reservar su posición.",
        ],
        "compraventa_fraude": [
            "Tengo otro comprador listo para pagar hoy, necesito su respuesta {urgencia_frase}.",
            "El artículo solo lo aparto {urgencia_frase} con pago confirmado.",
            "Si no me confirma {urgencia_frase}, lo vendo al siguiente interesado.",
            "El precio especial solo aplica {urgencia_frase}, después regresa al precio normal.",
        ],
        "_default": [
            "Debe resolver esto inmediatamente o perderá acceso.",
            "Tiene muy poco tiempo, actúe ahora para evitar consecuencias.",
            "El plazo vence hoy, no puede esperar más.",
        ],
    },

    # ------------------------------------------------------------------
    # CREDENTIAL_REQUEST — solicita datos sensibles
    # ------------------------------------------------------------------
    "credential_request": {
        "phishing_bancario": [
            "Para desbloquear su cuenta necesito que me proporcione su {credencial} de {banco}.",
            "El sistema requiere que confirme su {credencial} para proceder con la verificación.",
            "Ingrese su {credencial} en el enlace {url} para validar su identidad.",
            "Para continuar, proporcione su {credencial} — esta información es completamente segura.",
        ],
        "sat_devolucion": [
            "Para procesar su devolución necesito confirmar su {credencial}.",
            "Ingrese su {credencial} en el portal {url} para validar su expediente.",
            "El sistema del SAT requiere que autentique con su {credencial} para liberar los fondos.",
            "Proporcione su {credencial} para que podamos procesar su reembolso de {monto}.",
        ],
        "paqueteria_falsa": [
            "Para liberar el paquete necesito que confirme su {credencial}.",
            "Ingrese los datos de pago en {url} para cubrir los {monto} de arancel.",
            "Proporcione los datos de su tarjeta para el pago de {monto} de almacenaje.",
            "Necesito su {credencial} para registrar el pago y liberar su paquete.",
        ],
        "premio_oxxo": [
            "Para depositar su premio de {monto} necesito su {credencial}.",
            "Ingrese su {credencial} en {url} para que podamos procesar la entrega del premio.",
            "El departamento de premios de {empresa} necesita su {credencial} para la transferencia.",
            "Proporcione su {credencial} para que Recursos Humanos realice el depósito de {monto}.",
        ],
        "solicitud_otp": [
            "Por favor comparta el {credencial} que acaba de recibir en su teléfono.",
            "¿Cuál es el {credencial} que llegó a su celular? Lo necesito para cancelar la operación.",
            "El código OTP que recibió por SMS es el {credencial} que necesito para proteger su cuenta.",
            "Dígame el {credencial} que le enviamos por mensaje de texto.",
        ],
        "empleo_falso": [
            "Para formalizar su contrato necesito su {credencial} para registrarlo en el sistema.",
            "Envíe su {credencial} al correo de RR.HH. para completar el expediente.",
            "El sistema corporativo de {empresa} requiere su {credencial} para crear su perfil de empleado.",
            "Suba su {credencial} al formulario en {url} para continuar con el proceso.",
        ],
        "compraventa_fraude": [
            "Para hacer el pago de {monto} necesito su {credencial} de depósito.",
            "Deme su {credencial} y le transfiero los {monto} de inmediato.",
            "El sistema de {plataforma} requiere su {credencial} para procesar la compra.",
            "Mándeme su {credencial} y le hago la transferencia por los {monto} acordados.",
        ],
        "_default": [
            "Necesito que me proporcione su información de acceso para continuar.",
            "Por favor comparta sus credenciales para verificar su identidad.",
            "El sistema requiere sus datos para procesar su solicitud.",
        ],
    },

    # ------------------------------------------------------------------
    # PAYMENT_PRESSURE — exige transferencia o depósito
    # ------------------------------------------------------------------
    "payment_pressure": {
        "phishing_bancario": [
            "Hay un cargo pendiente de {monto} que debe liquidar para reactivar su {servicio}.",
            "Para completar la verificación debe realizar un pago de {monto} como garantía.",
            "El proceso de desbloqueo tiene un costo administrativo de {monto} en {banco}.",
            "Transfiera {monto} a la cuenta de seguridad de {banco} para recuperar el acceso.",
        ],
        "sat_devolucion": [
            "Debe pagar {monto} de trámite para liberar su devolución de impuestos.",
            "El proceso de reembolso tiene un cargo fiscal de {monto} que debe liquidar.",
            "Para recibir su devolución de {monto} debe primero pagar {monto} de gastos notariales.",
            "Hay un arancel de {monto} para procesar su reembolso en {entidad}.",
        ],
        "paqueteria_falsa": [
            "El pago de {monto} debe realizarse {urgencia_frase} para liberar su paquete.",
            "Transfiera {monto} a la cuenta de {paqueteria} para el despacho aduanal.",
            "El cargo de {monto} debe cubrirse en efectivo o transferencia bancaria.",
            "Para liberar su paquete realice el pago de {monto} en {url}.",
        ],
        "premio_oxxo": [
            "Para recibir su premio de {monto} debe pagar {monto} de impuestos por sorteo.",
            "El ganador es responsable del impuesto sobre premios equivalente a {monto}.",
            "Para que le depositemos en {monto}, debe primero cubrir los impuestos de {monto}.",
            "La ley exige que el ganador pague {monto} de ISR antes de recibir el premio.",
        ],
        "solicitud_otp": [
            "Para cancelar la transferencia no autorizada de {monto} debe realizar un contracargo de {monto}.",
            "El sistema requiere un depósito de verificación de {monto} para cancelar la operación.",
            "Para proteger su cuenta realice una transferencia de {monto} a la cuenta de seguridad.",
            "Bloquee el cargo enviando {monto} a la cuenta de reversión del banco.",
        ],
        "empleo_falso": [
            "Para iniciar realice el depósito de {monto} por el kit de trabajo remoto.",
            "El contrato se firma después de confirmar el pago de {monto} de membresía.",
            "El primer mes de herramientas tiene costo de {monto}, se descuenta del primer sueldo.",
            "Transfiera {monto} a la cuenta corporativa de {empresa} para procesar su alta.",
        ],
        "compraventa_fraude": [
            "Para apartar el artículo envíe {monto} de anticipo a mi cuenta.",
            "El total es {monto}, necesito el pago completo antes de enviar.",
            "Mándeme los {monto} y le envío el artículo con número de guía.",
            "Realice la transferencia de {monto} a mi CLABE y le confirmo el envío.",
        ],
        "_default": [
            "Necesita realizar un pago para continuar con el proceso.",
            "El cargo debe liquidarse hoy para evitar consecuencias mayores.",
            "Transfiera el monto indicado para resolver su situación.",
        ],
    },

    # ------------------------------------------------------------------
    # THREAT_ESCALATE — amenaza con consecuencias graves
    # ------------------------------------------------------------------
    "threat_escalate": {
        "phishing_bancario": [
            "Si no actúa {urgencia_frase}, procederemos al bloqueo permanente e irreversible de su cuenta en {banco}.",
            "El departamento legal de {banco} iniciará un proceso de investigación por fraude si no verifica.",
            "Sin verificación {urgencia_frase}, su historial crediticio en {banco} podría verse afectado.",
            "La cuenta será cancelada y el saldo congelado si no completamos la verificación {urgencia_frase}.",
        ],
        "sat_devolucion": [
            "El SAT podría iniciar una auditoría a su RFC si no regulariza su situación {urgencia_frase}.",
            "Si no completa el trámite {urgencia_frase}, {entidad} emitirá un requerimiento formal.",
            "Ignorar este proceso podría resultar en multas y recargos sobre su declaración.",
            "El incumplimiento puede derivar en un crédito fiscal a cargo y embargo de bienes.",
        ],
        "paqueteria_falsa": [
            "Si no paga {urgencia_frase}, el paquete será destruido y no habrá reembolso.",
            "Los cargos de almacenaje aumentan cada día, ya suma {monto} y sigue creciendo.",
            "Sin el pago de {monto}, el contenido será donado a instituciones de beneficencia.",
            "Procederemos al remate del contenido si no regulariza {urgencia_frase}.",
        ],
        "premio_oxxo": [
            "Si no reclama {urgencia_frase}, el premio de {monto} se asignará automáticamente al suplente.",
            "Los premios no reclamados en plazo son donados a fundaciones benéficas de {empresa}.",
            "La empresa {empresa} no puede mantener el premio reservado más tiempo sin confirmación.",
            "Perderá definitivamente el derecho al premio si no actúa {urgencia_frase}.",
        ],
        "solicitud_otp": [
            "Si no proporciona el código {urgencia_frase}, la transferencia de {monto} se completará.",
            "Sin el OTP, no podemos detener el cargo y será muy difícil recuperar los {monto}.",
            "El sistema procesará la transacción automáticamente si no intervenimos {urgencia_frase}.",
            "Después de ese tiempo no será posible cancelar el cargo de {monto} a su cuenta.",
        ],
        "empleo_falso": [
            "Si no realiza el pago {urgencia_frase}, la plaza será asignada al siguiente candidato.",
            "La empresa tomará la falta de pago como desinterés y cerrará su expediente.",
            "Perderá la oportunidad de ganar {monto} mensuales si no actúa de inmediato.",
            "Sin confirmación {urgencia_frase}, el contrato se ofrecerá a otro candidato.",
        ],
        "compraventa_fraude": [
            "Si no confirma el pago {urgencia_frase}, lo vendo al otro comprador que ya está esperando.",
            "Ya tengo tres personas más interesadas, si no paga hoy no puedo guardarle el artículo.",
            "El precio sube mañana si no cerramos hoy, aproveche que sigue en {monto}.",
            "Necesito certeza de compra, si no hay pago hoy cancelo la reserva sin excepción.",
        ],
        "_default": [
            "Las consecuencias de no actuar serán muy graves y difíciles de revertir.",
            "Sin su cooperación inmediata el problema se agravará significativamente.",
            "El tiempo se acaba y las consecuencias son su responsabilidad.",
        ],
    },

    # ------------------------------------------------------------------
    # RECOVERY_OFFER — ofrece la "solución" (cierre del fraude)
    # ------------------------------------------------------------------
    "recovery_offer": {
        "phishing_bancario": [
            "La solución es simple: ingrese a {url} con su {credencial} y el sistema lo desbloqueará.",
            "En cuanto confirme su {credencial} en {url} su cuenta quedará reactivada en minutos.",
            "Solo necesita llamar al {phone} con su {credencial} listo y resolvemos esto en 5 minutos.",
            "El proceso de recuperación es rápido: visite {url} y siga las instrucciones con su {credencial}.",
        ],
        "sat_devolucion": [
            "Ingrese al portal {url} con su {credencial} y la devolución se liberará en 24 horas.",
            "El trámite es sencillo: llame al {phone} con su {credencial} para iniciar la transferencia.",
            "Complete el formulario en {url} con su {credencial} y recibirá su devolución pronto.",
            "Solo necesita hacer un depósito de {monto} para cubrir gastos y recibir su reembolso.",
        ],
        "paqueteria_falsa": [
            "Realice el pago de {monto} en {url} y su paquete estará disponible para entrega.",
            "Llame al {phone} con su tarjeta lista para pagar los {monto} y coordinamos la entrega.",
            "El pago de {monto} en efectivo en cualquier OXXO usando la referencia que le damos libera el paquete.",
            "Una vez confirmado el pago en {url}, la entrega se programa para el siguiente día hábil.",
        ],
        "premio_oxxo": [
            "Solo necesita depositar {monto} de impuestos y le transferimos su premio de {monto}.",
            "Complete el registro en {url} con su {credencial} y recibe el depósito en 24 horas.",
            "Llame al {phone} para confirmar sus datos y programar la entrega del premio.",
            "El proceso es: paga {monto} de ISR → le depositamos {monto} del premio en 48 horas.",
        ],
        "solicitud_otp": [
            "Proporcione el {credencial} ahora y cancelamos el cargo inmediatamente.",
            "Dígame el código y en segundos bloqueamos la transacción antes de que se procese.",
            "Con el {credencial} podemos revertir todo y su cuenta quedará segura.",
            "El código OTP es la única forma de detener el cargo de {monto}, compártalo ahora.",
        ],
        "empleo_falso": [
            "Deposite los {monto} del kit y le enviamos el contrato firmado por {empresa} de inmediato.",
            "Una vez confirmado su pago le hacemos llegar el equipo de trabajo y el contrato.",
            "El pago se descuenta del primer salario, es solo un trámite administrativo de {empresa}.",
            "Confirme el depósito de {monto} al correo de RR.HH. y iniciamos el proceso de alta.",
        ],
        "compraventa_fraude": [
            "Haga el depósito a mi CLABE y le mando foto del envío con número de guía.",
            "Pague los {monto} por transferencia y en cuanto confirme hago el paquete.",
            "Le doy el número de cuenta para que transfiera y coordino el envío hoy mismo.",
            "Acuerdo: usted paga, yo envío. Comparta su dirección y hago el paquete.",
        ],
        "_default": [
            "La solución es rápida: complete el trámite en el enlace que le enviamos.",
            "Con su información podremos resolver todo en minutos.",
            "Solo necesita seguir las instrucciones y todo quedará arreglado.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Templates para conversaciones legítimas
# ---------------------------------------------------------------------------

LEGIT_TEMPLATES: dict[str, list[str]] = {
    "trabajo": [
        "Hola, ¿ya terminaste el informe del proyecto?",
        "La junta del viernes se cambió a las 3 pm, ¿puedes?",
        "El cliente confirmó la reunión para la próxima semana.",
        "Necesito el reporte de ventas antes del jueves.",
        "¿Revisaste los comentarios que dejé en el documento?",
        "El equipo estará disponible para la llamada a las 11.",
        "Recuerda que mañana es la presentación con el cliente.",
        "¿Puedes enviarme el archivo de Excel que trabajamos ayer?",
        "La entrega del proyecto es el lunes, ¿vamos bien?",
        "Gracias por el apoyo en la propuesta, quedó muy bien.",
    ],
    "personal": [
        "Hola, ¿cómo estás? Hace tiempo que no sabemos nada de ti.",
        "¿Pudiste ver el partido anoche? Increíble el resultado.",
        "Nos vemos el sábado para la cena, ¿confirmas?",
        "Feliz cumpleaños, espero que tengas un excelente día.",
        "¿Ya solucionaste el problema con el carro?",
        "El clima está muy extraño estos días, ¿verdad?",
        "¿Fuiste al médico? Espero que ya estés mejor.",
        "La película que me recomendaste estuvo muy buena.",
        "¿Cómo les fue en el viaje? Ya queremos ver las fotos.",
        "Avísame cuando llegues para saber que estás bien.",
    ],
    "negocios": [
        "Le comparto la cotización que solicitó, quedo en espera de sus comentarios.",
        "Adjunto la factura correspondiente al servicio del mes pasado.",
        "El pedido fue procesado y saldrá mañana por la mañana.",
        "Confirmo recepción de su pago, muchas gracias.",
        "¿Podría compartirme sus datos de facturación para emitir la factura?",
        "La mercancía llegó en perfectas condiciones, muchas gracias.",
        "Le envío el contrato de servicio para su revisión y firma.",
        "¿Le funciona el martes a las 10 para revisar la propuesta?",
        "Quedamos de acuerdo en los términos discutidos, ¿correcto?",
        "El técnico estará en su oficina el jueves entre 9 y 11.",
    ],
    "familiar": [
        "Mamá, ¿ya llegaste? Avísame.",
        "¿Cómo están los niños? Mándale saludos al abuelo.",
        "El domingo nos juntamos en casa de la abuela, ¿van?",
        "¿Ya confirmaste si va a venir tu hermana a la posada?",
        "Te mando el dinero de la renta el lunes, ¿está bien?",
        "¿Qué quieres de regalo de cumpleaños? Ya se acerca la fecha.",
        "Papá dice que el carro ya está listo, puedes pasar por él.",
        "¿Llevas algo a la reunión o vamos todos a manos vacías?",
        "La comida del domingo es en casa de los tíos, no aquí.",
        "Llámame cuando puedas, tengo algo que contarte.",
    ],
    "compras": [
        "¿Ya llegó tu pedido de Amazon? El mío tardó tres días.",
        "Vi que hay descuentos en Liverpool esta semana.",
        "El producto que pedí llegó en mal estado, lo tuve que devolver.",
        "¿Sabes si el OXXO de la esquina tiene servicio de pago de servicios?",
        "Encontré el libro que buscabas en Gandhi, ¿lo aparto?",
        "El supermercado tiene oferta 2x1 en bebidas esta semana.",
        "¿Ya fuiste a ver los muebles que te recomendé?",
        "Mi pedido llegó completo y bien empaquetado, muy buen servicio.",
        "¿Dónde compraste ese electrodoméstico? Me gustó mucho.",
        "El precio bajó en MercadoLibre, ya está en lo que buscabas.",
    ],
}
