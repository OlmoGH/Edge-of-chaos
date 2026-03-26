# Análisis de resultados

## Estudio de las oscilaciones

La velocidad de los autovalores de la matriz $W$ se pueden obtener mediante la siguiente expresión

$$
\dot{\lambda} = y^*\dot{W}x
$$

sustituyendo $\dot{W}$ en la ecuación y simplificando obtenemos

$$
\dot{\lambda} = \alpha (1 - (v^*\cdot x)(x^T \cdot u)) = \alpha (1-\braket{v | x}  \braket{x|u})
$$

Volvemos a derivar con respecto a $t$ y simplificamos

$$
\ddot{\lambda} = -4\alpha\braket{u|x}\braket{x|v}\Re{\lambda}
$$