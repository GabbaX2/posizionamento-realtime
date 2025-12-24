# Sistema di Validazione - Changelog

---

## v3.0 - Overlay Semi-Trasparente + Rilevamento Cerchi (24 Dicembre 2024)

### Descrizione
Nuovo approccio di validazione basato su:
1. **Overlay Reference**: Il disegno include la **gamba con ~15% opacity** (non solo il contorno)
2. **Template Matching**: Confronto template per verificare allineamento zona
3. **HoughCircles**: Rileva **cerchi fisici** con HoughCircles e verifica se sono **concentrici** ai cerchi del disegno

### Modifiche Reference Processor (`processors/reference_processor.py`)

**Nuovo sistema overlay BGRA:**
- Crea overlay 4 canali (BGRA) invece di semplice maschera BGR
- La gamba viene inclusa con 15% di opacità per template matching
- Contorno gamba, cerchi sensori e testo hanno alpha pieno (255)
- Nuovo metodo `_to_b64_rgba()` per esportare PNG con canale alpha

**Nuovo output:**
```python
{
    'drawing_overlay_b64': '...',  # Nuovo: PNG con alpha
    'sensor_circles': [             # Nuovo: Lista flat sensori
        {'id': 1, 'x': 100, 'y': 150, 'r': 30, 'group': 1},
        ...
    ]
}
```

### Modifiche Validation Service (`services/validation_service.py`)

**Nuovi metodi:**

1. `_check_zone_alignment(live_frame, reference_overlay)`:
   - Template matching per verificare posizionamento gamba
   - Usa finestra centrale del reference come template
   - Soglia: 0.55 (55% correlazione) per lock

2. `_detect_and_verify_sensors(live_frame, sensor_circles_ref)`:
   - HoughCircles per rilevare cerchi nel frame live
   - Verifica concentricità: distanza centri < 25px
   - Restituisce stato per ogni sensore di riferimento

3. `_generate_group_feedback(sensor_results, steps_config, current_step)`:
   - Genera feedback specifico per gruppo di sensori
   - "Manca 1 sensore nel gruppo" / "Mancano N sensori, posizionarli"

**Nuova risposta API:**
```json
{
    "success": true,
    "accuracy": 85.0,
    "is_locked": true,
    "zone_match": 68.5,
    "message": "Zona OK",
    "direction": "Manca 1 sensore nel gruppo",
    "group_feedback": "Manca 1 sensore nel gruppo",
    "sensors_status": [...]
}
```

### Modifiche Frontend (`templates/validation.html`)

**Fase NON Locked:**
- Overlay con opacity 50% (maggiore per vedere la gamba)
- Cornice arancione tratteggiata
- Label: "🔴 SOVRAPPONI LA GAMBA ALL'OVERLAY"
- Mostra percentuale match zona

**Fase Locked:**
- Cornice verde solida con ombra
- Badge: "✅ ZONA VALIDATA - {group_feedback}"
- Sensori mancanti: cerchi rossi pulsanti con "?"
- Sensori trovati: cerchi verdi con ombra

---

# Miglioramenti Sistema di Validazione - Contorno Preciso

## Panoramica
Il sistema di validazione è stato completamente rinnovato per utilizzare un confronto preciso dei contorni invece di un semplice allineamento percentuale. Questo garantisce che il disegno sia effettivamente allineato e non storto.

## Modifiche Implementate

### 1. Backend (`services/validation_service.py`)

#### Nuovi Metodi:

**`_extract_reference_contour(ref_mask)`**
- Estrae il contorno preciso dal disegno di riferimento caricato
- Usa edge detection + findContours per ottenere la forma esatta
- Semplifica leggermente il contorno per renderlo robusto al matching
- Salva il contorno nella sessione per confronti futuri

**`_extract_live_contour(live_img)`**
- Estrae il contorno dalla camera live in tempo reale
- Preprocessing robusto con:
  - GaussianBlur per ridurre rumore
  - Canny Edge Detection
  - Operazioni morfologiche (closing) per connettere bordi
- Filtro di centralità: prende solo contorni al centro dello schermo
- Restituisce sia l'oggetto contorno che i punti per il frontend

**`_calculate_visual_alignment_score(live_contour, ref_contour, dims)`**
- Calcola uno score visivo 0-100 basato su IoU (Intersection over Union)
- Usato per dare feedback progressivo all'utente durante l'allineamento
- Non è il criterio principale di lock, ma serve per la UX

#### Sistema di Validazione a 3 Fasi:

**FASE 1: Estrazione Contorno Live**
- Estrae il contorno dalla camera in tempo reale
- Pulizia e preprocessing per gestire rumore ambientale

**FASE 2: Shape Matching (NUOVO!)**
- Usa `cv2.matchShapes()` per confrontare i contorni
- Metrica: CV_CONTOURS_MATCH_I1 (invariante a scala/rotazione/traslazione)
- Soglia: shape_similarity < 0.15 per considerare i contorni simili
- Calcola anche alignment_score visivo per feedback UX

**FASE 3: Verifica Sensori (solo se locked)**
- Il lock si attiva SOLO se:
  - `shape_similarity < 0.15` (contorni molto simili)
  - `alignment_score >= 75.0` (buona sovrapposizione visiva)
- I sensori vengono verificati SOLO se sono dentro il contorno validato
- Usa `cv2.pointPolygonTest()` per verificare se un sensore è nell'area

#### Nuovo Response API:
```json
{
  "success": true,
  "accuracy": 85.0,
  "is_locked": true,
  "live_contour": [[x1,y1], [x2,y2], ...],
  "shape_match": 92.5,  // NUOVO: % similarità forma (0-100)
  "message": "Contorno OK",
  "direction": "Posiziona sensori: 3/5",
  "sensors_status": [
    {
      "id": "s1",
      "step": 1,
      "present": true,
      "in_area": true  // NUOVO: indica se il sensore è nell'area validata
    }
  ]
}
```

### 2. Frontend (`templates/validation.html`)

#### Rendering Migliorato:

**Fase NON Locked (Allineamento):**
- Overlay di riferimento semi-trasparente (35% opacity)
- Contorno di riferimento: linea tratteggiata verde come guida
- Contorno live: linea rossa spessa (4px) con ombra che segue la gamba
- Label chiara: "🔴 ALLINEA IL CONTORNO ROSSO"
- Istruzioni: "Sovrapponi al disegno verde"

**Fase Locked (Validazione):**
- Cornice verde SOLIDA spessa (6px) con ombra luminosa
- Badge di conferma: "✅ CONTORNO VALIDATO - SCANSIONE..."
- Sensori visualizzati come cerchi:
  - Verde brillante + ombra se trovati
  - Oro pulsante se mancanti
  - Solo i sensori nell'area validata vengono controllati

#### Nuove Variabili JavaScript:
```javascript
let shapeMatchScore = 0; // % similarità forma dal backend
```

#### Feedback Utente Migliorato:

**Durante l'allineamento:**
- "Inquadra la gamba" (< 10% accuracy)
- "Avvicinati... Contorno rilevato - allinealo (match: XX%)" (10-50%)
- "Quasi... Centra e sovrapponi il disegno" (50-75%)

**Dopo il lock:**
- "Contorno Validato - Attesa Config Sensori" (nessun sensore da trovare)
- "Contorno OK - Posiziona sensori: N/M" (sensori parziali)
- "Perfetto - Step Completato!" (tutti i sensori trovati)

## Vantaggi del Nuovo Sistema

1. **Precisione**: Il confronto shape matching è robusto a piccole variazioni
2. **No False Positive**: Non si blocca su allineamenti casuali o storti
3. **Feedback Chiaro**: L'utente vede esattamente cosa deve fare
4. **Area Controllata**: I sensori vengono cercati solo nell'area validata
5. **UX Migliorata**: Visualizzazione chiara con colori e animazioni

## Soglie e Parametri

- **Shape Similarity Threshold**: 0.15 (più basso = più preciso)
- **Alignment Score Threshold**: 75.0 (per feedback visivo)
- **Lock Condition**: `(shape_similarity < 0.15) AND (alignment_score >= 75.0)`
- **Sensor Area Check**: `cv2.pointPolygonTest(live_contour, point, False) >= 0`

## Prossimi Miglioramenti Possibili

1. Regolazione dinamica delle soglie in base all'ambiente
2. Salvataggio del contorno validato per confronti step multipli
3. Visualizzazione della percentuale di shape match in UI
4. Feedback audio quando il lock è attivo
5. Sistema di auto-calibrazione della soglia shape matching

## Testing

Per testare il nuovo sistema:
1. Carica un'immagine di riferimento
2. Avvia la validazione
3. Osserva la linea rossa che segue la gamba
4. Allinea il contorno rosso con il disegno verde tratteggiato
5. Quando il lock si attiva, vedrai la cornice verde solida
6. Posiziona i sensori nell'area validata
