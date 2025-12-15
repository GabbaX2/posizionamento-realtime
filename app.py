from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import base64
import logging
import io
from PIL import Image

# Import CONFIG
from config import Config

# NUOVI IMPORT CORRETTI
from services.validation_service import ValidationService
from utils.background_removal import remove_background_from_pil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Inizializza il servizio di validazione
validation_service = ValidationService()


# ============================================================================
# ROUTES - PAGINE HTML
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/validation')
def validation_page():
    return render_template('validation.html')

@app.route('/login')
def login():
    return render_template('login.html')


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload_reference', methods=['POST'])
def api_upload_reference():
    """
    Endpoint per caricare un'immagine di riferimento.
    Delega tutto al ValidationService -> ReferenceProcessor.
    """
    try:
        logger.info("[ROUTE] /api/upload_reference called")

        result = validation_service.handle_reference_upload(request)

        if result.get('success'):
            # Calcoliamo il numero di sensori per il log (dato che la struttura è cambiata)
            steps_config = result.get('steps_config', {})
            sensors_count = sum(len(v) for v in steps_config.values())

            logger.info("✅ Upload e processing completato")
            logger.info(f"   🔬 Sensori rilevati: {sensors_count}")
        else:
            logger.warning(f"⚠️ Upload fallito: {result.get('error', 'Unknown error')}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Route /api/upload_reference failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Errore server: {str(e)}'
        }), 500


@app.route('/api/remove_background', methods=['POST'])
def api_remove_background():
    """
    Endpoint di utility per rimuovere lo sfondo (es. per test frontend).
    """
    try:
        logger.info("[ROUTE] /api/remove_background called")
        data = request.get_json()

        if not data or 'image_base64' not in data:
            return jsonify({'success': False, 'error': 'Parametro image_base64 mancante'}), 400

        image_base64 = data.get('image_base64')

        # Decodifica immagine
        try:
            if isinstance(image_base64, str) and ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]
            image_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({'success': False, 'error': 'Immagine non valida'}), 400

        # --- MODIFICA IMPORTANTE QUI ---
        # Usiamo la firma semplificata definita in utils/background_removal.py
        # Non passiamo più resize_max o grabcut_if_failed perché sono gestiti internamente
        out_pil, method = remove_background_from_pil(pil_img)

        if out_pil is not None:
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            result_image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            logger.info(f"✅ Sfondo rimosso con successo usando {method}")

            return jsonify({
                'success': True,
                'image_base64': result_image,
                'method': method,
                'message': 'Sfondo rimosso con successo'
            })
        else:
            logger.warning("⚠️ Rimozione sfondo fallita")
            return jsonify({'success': False, 'error': 'Impossibile rimuovere lo sfondo'}), 500

    except Exception as e:
        logger.error(f"[ERROR] Route /api/remove_background failed: {e}")
        return jsonify({'success': False, 'error': f'Errore server: {str(e)}'}), 500


@app.route('/api/validate_frame', methods=['POST'])
def api_validate_frame():
    """
    Endpoint per validare un frame dalla webcam (Live Check)
    """
    try:
        # Questo chiama il metodo aggiornato "Edge-Based" del servizio
        result = validation_service.validate_current_frame(request)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[ERROR] Route /api/validate_frame failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Errore server: {str(e)}'
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================================
# ERROR HANDLERS E MAIN
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint non trovato'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"[ERROR 500] {error}")
    return jsonify({'success': False, 'error': 'Errore interno del server'}), 500


if __name__ == '__main__':
    # Setup cartelle
    os.makedirs('uploads', exist_ok=True)
    # Rimuoviamo cartelle vecchie non più usate dal nuovo approccio

    print("\n🚀 Avvio applicazione Flask...")
    app.run(host='0.0.0.0', port=5000)
