# 🔋 Li-ion Battery Prognosis Model

Este repositorio implementa un **modelo físico electroquímico de bajo orden** para una celda Li-ion, junto con un **Unscented Kalman Filter (UKF)** para estimación en línea de estados internos no medibles. El objetivo es facilitar simulaciones de descarga, diagnosis y prognosis de celdas usando datos reales.

---

## 📁 Estructura del Proyecto

**Directorio principal:** `Model/`

```bash
Model/
├── BatteryData.py
├── BatteryModels.py
├── BatteryParameters.py
├── testBatteryModel.py
├── testUKF.py
├── UnscentedKalmanFilter.py
├── utils.py
└── requirements.txt
```

## 📁 Descripción de los módulos

- **`BatteryData.py`** — Carga y procesamiento de datos de corriente y voltaje.
- **`BatteryModels.py`** — Implementación del modelo físico electroquímico.
- **`BatteryParameters.py`** — Definición de parámetros físico-químicos.
- **`UnscentedKalmanFilter.py`** — Algoritmo UKF para estimación de estados.
- **`utils.py`** — Funciones para generar gráficos de simulación y resultados.
- **`testBatteryModel.py`** — Simulación del modelo físico con datos de entrada.
- **`testUKF.py`** — Ejecución del UKF con datos reales.
- **`requirements.txt`** — Lista de dependencias necesarias.

---

## 🚀 Cómo ejecutar

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu_usuario/Li-ion-Battery-Prognosis.git
   cd Li-ion-Battery-Prognosis/Model

## 🚀 Cómo ejecutar

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu_usuario/Li-ion-Battery-Prognosis.git
   cd Li-ion-Battery-Prognosis/Model
   ```

2. *(Opcional pero recomendado)* Crea y activa un entorno virtual:

   ```bash
   python -m venv .venv
   # Activar en Linux/macOS
   source .venv/bin/activate
   # Activar en Windows
   .venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta la estimación con UKF:

   ```bash
   python testUKF.py
   ```

Asegúrate de tener los archivos de datos de corriente y voltaje en las carpetas:

- /data/RW9_Current_Discharge_Reference/
- /data/RW9_Voltage_Discharge_Reference/

---

## Ejecutar las simulaciones del modelo de batería

El script `testBatteryModel.py` permite correr distintas simulaciones seleccionando el tipo mediante el argumento `--simulation`. La carpeta `imgs/` donde se guardan los gráficos se crea automáticamente si no existe.

### Comando general para ejecutar:

```bash
python testBatteryModel.py --simulation <tipo>
```

### Tipos de simulación disponibles:

| Valor | Descripción                                      |
|-------|------------------------------------------------|
| 1     | Simulación con carga constante                   |
| 2     | Simulación con carga pulsada                      |
| 3     | Simulación con corte por voltaje mínimo (cut-off)|
| 4     | Comparación con descarga de referencia            |

### Ejemplos de uso:

- Simulación con carga constante:

  ```bash
  python testBatteryModel.py --simulation 1
  ```

- Simulación con carga pulsada:

  ```bash
  python testBatteryModel.py --simulation 2
  ```

- Simulación con corte por voltaje mínimo:

  ```bash
  python testBatteryModel.py --simulation 3
  ```

- Comparación con descarga de referencia:

  ```bash
  python testBatteryModel.py --simulation 4
  ```

### Salida

- Los gráficos se guardan automáticamente en la carpeta `imgs/` como archivos PDF (por ejemplo, `imgs/battery_discharge_profile.pdf`).
- Si la carpeta `imgs/` no existe, el script la crea automáticamente.

---

## 📊 Datos de referencia

Los datos reales de descarga y perfiles de corriente/voltaje se encuentran organizados en:

- /data/RW9_Current_Discharge_Reference/
- /data/RW9_Voltage_Discharge_Reference/

Cada archivo CSV corresponde a un ciclo individual de descarga.

---

## 📄 Referencias

Este proyecto se basa en el trabajo de:

Daigle, M., & Kulkarni, C. (2013). *Electrochemistry-based battery modeling for prognostics*.  
_Annual Conference of the Prognostics and Health Management Society._