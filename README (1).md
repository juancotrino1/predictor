# BTC 15-min Predictor — Render Deploy

## Estructura del proyecto
```
btc-render/
├── app.py              ← predictor + Flask web server
├── requirements.txt
├── render.yaml
└── templates/
    └── index.html      ← dashboard con logs en vivo
```

## Cómo desplegar (5 pasos)

### 1. Sube el código a GitHub
```bash
git init
git add .
git commit -m "btc predictor"
# Crea un repo en github.com y luego:
git remote add origin https://github.com/TU_USUARIO/btc-predictor.git
git push -u origin main
```

### 2. Crea una cuenta en Render
Ve a https://render.com y regístrate (gratis).

### 3. Nuevo Web Service
- Dashboard → **New +** → **Web Service**
- Conecta tu repositorio de GitHub
- Render detecta `render.yaml` automáticamente, o configura:
  - **Runtime:** Python 3
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `gunicorn app:app --workers 1 --threads 4 --timeout 0 --bind 0.0.0.0:$PORT`

### 4. Deploy
Haz clic en **Create Web Service**. El primer build tarda ~5-10 min
(instala PyTorch + TensorFlow).

### 5. Abre el dashboard
Render te da una URL del tipo `https://btc-predictor.onrender.com`.
Ábrela y verás el dashboard con logs en tiempo real.

---

## Qué hace el dashboard

| Sección | Descripción |
|---|---|
| **Tarjetas superiores** | Última predicción de cada modelo (precio + dirección) |
| **Tabla historial** | Todas las predicciones guardadas en memoria |
| **Log en vivo** | Stream en tiempo real del proceso (fetch → entrenamiento → resultado) |

---

## Notas importantes

- **Free tier de Render** duerme tras 15 min de inactividad → la primera visita
  puede tardar ~30 s en despertar. Usa el plan **Starter ($7/mes)** para
  que siempre esté activo.
- Los datos se guardan **en memoria** (no hay disco persistente en free tier).
  Al reiniciar el servicio se pierde el historial. Si necesitas persistencia,
  añade una base de datos PostgreSQL gratuita en Render.
- PyTorch + TensorFlow consumen ~1.5 GB de RAM. El free tier tiene 512 MB,
  por lo que **se recomienda el plan Starter (512 MB → 2 GB)**.
