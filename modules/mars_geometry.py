import math

# Radio medio de Marte en kilómetros
MARS_RADIUS_KM = 3389.5


def mars_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia de círculo máximo entre
    dos coordenadas sobre una esfera marciana.

    Parámetros:
        lat1, lon1: coordenadas iniciales en grados
        lat2, lon2: coordenadas finales en grados

    Retorna:
        Distancia en kilómetros.
    """

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return MARS_RADIUS_KM * c