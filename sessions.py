"""Almacén de sesiones de chat en memoria.

Cada sesión guarda el nivel activo (1 o 2) y el historial de turnos. Cuando una
consulta se escala —porque el nivel 1 no puede resolverla o porque el usuario
marca insatisfacción— la sesión sube a nivel 2 y se mantiene ahí.

NOTA: es un dict en memoria; las sesiones se pierden al reiniciar el servidor.
En producción se usaría Redis o una base de datos.
"""

import uuid
from typing import Optional

# session_id -> datos de la sesión
SESIONES: dict[str, dict] = {}


def crear_sesion() -> str:
    """Crea una sesión nueva en nivel 1 y devuelve su identificador."""
    session_id = uuid.uuid4().hex
    SESIONES[session_id] = {
        "historial": [],  # lista de turnos {pregunta, respuesta, nivel}
        "activa": True,
    }
    return session_id


def get_sesion(session_id: str) -> Optional[dict]:
    return SESIONES.get(session_id)


def registrar_turno(session_id: str, pregunta: str, respuesta: str) -> None:
    sesion = SESIONES.get(session_id)
    if sesion is not None:
        sesion["historial"].append(
            {"pregunta": pregunta, "respuesta": respuesta}
        )


def finalizar(session_id: str) -> Optional[dict]:
    """Cierra la sesión (el usuario quedó a gusto) y devuelve sus datos."""
    sesion = SESIONES.get(session_id)
    if sesion is not None:
        sesion["activa"] = False
    return sesion