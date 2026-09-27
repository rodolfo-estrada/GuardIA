# GuardIA 🛡️

Sistema inteligente de videovigilancia orientado al monitoreo, gestión de alertas y análisis de amenazas mediante una arquitectura web con componentes de inteligencia artificial.

## Descripción

GuardIA integra un panel web para administración y monitoreo con servicios backend y un módulo de Machine Learning. El proyecto está diseñado para centralizar cámaras, usuarios, alertas, registros y análisis de riesgo dentro de una misma plataforma.

## Tecnologías principales

- React
- TypeScript
- Vite
- Tailwind CSS
- Node.js
- Express
- PostgreSQL
- Firebase / Firebase Admin
- Python
- FastAPI
- scikit-learn
- Docker

## Funcionalidades

- Autenticación y control de acceso por roles
- Gestión de usuarios
- Gestión de cámaras
- Monitoreo y visualización de alertas
- Registro de actividad e incidentes
- API backend con Express
- Integración con PostgreSQL
- Integración con Firebase
- Servicio de Machine Learning para clasificación de nivel de amenaza
- Comparación de modelos Random Forest, Gradient Boosting y MLP
- Panel de análisis de Machine Learning

## Arquitectura

El proyecto se divide principalmente en:

- `src/`: frontend React + TypeScript
- `api/`: API backend con Node.js y Express
- `ml_service/`: entrenamiento e inferencia de modelos de Machine Learning
- `db/`: scripts de inicialización y datos de desarrollo
- `docs/`: documentación técnica
- `architecture/`: diagramas de arquitectura
- `mockups/`: prototipos y referencias visuales

## Configuración

### Frontend

```bash
npm install
npm run dev
```

### Backend

```bash
cd api
npm install
npm run dev
```

### Servicio de Machine Learning

```bash
python ml_service/api.py
```

## Variables de entorno

El repositorio no incluye credenciales reales. Utiliza los archivos `.env.example` como referencia y crea tus propios archivos `.env` localmente.

Nunca publiques claves privadas, credenciales de Firebase ni cadenas de conexión de bases de datos.

## Estado del proyecto

Proyecto académico y de portafolio en desarrollo activo.

## Autor

**Rodolfo Estrada Sánchez**

GitHub: [rodolfo-estrada](https://github.com/rodolfo-estrada)
